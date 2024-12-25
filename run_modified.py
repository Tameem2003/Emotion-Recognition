import argparse
import numpy as np
import pandas as pd
import os
import time
from scipy import stats
from functions import sequences
from functions import get_face_areas
from functions.get_models import load_weights_EE, load_weights_LSTM
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Argument parsing setup
parser = argparse.ArgumentParser(description="Run Emotion Detection on Videos")
parser.add_argument('--path_video', type=str, default='video/', help='Path to all videos')
parser.add_argument('--path_save', type=str, default='report/', help='Path to save the report')
parser.add_argument('--conf_d', type=float, default=0.7, help='Elimination threshold for false face areas')
parser.add_argument('--path_FE_model', type=str, default='models_EmoAffectnet/weights_0_66_37_wo_gl.h5',
                    help='Path to a model for feature extraction')
parser.add_argument('--path_LSTM_model', type=str, default='models_EmoAffectnet/weights_0_66_49_wo_gl.h5',
                    help='Path to a model for emotion prediction')

args = parser.parse_args()

def fix_layer_names(name):
    """Fix layer names to avoid issues with custom layers."""
    return name.replace('/', '_')

def load_model_with_custom_names(filepath):
    """Load model with adjusted custom layer names."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
        
    try:
        custom_objects = {
            'Conv2D': lambda **kwargs: tf.keras.layers.Conv2D(
                **{k: v for k, v in kwargs.items() if k != 'name'} | 
                {'name': fix_layer_names(kwargs['name']) if 'name' in kwargs else None}
            ),
            'BatchNormalization': lambda **kwargs: tf.keras.layers.BatchNormalization(
                **{k: v for k, v in kwargs.items() if k != 'name'} | 
                {'name': fix_layer_names(kwargs['name']) if 'name' in kwargs else None}
            )
        }
        model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model from {filepath}: {e}")

def pred_one_video(path):
    """Predict emotions for one video."""
    os.makedirs(args.path_video, exist_ok=True)
    os.makedirs(args.path_save, exist_ok=True)

    start_time = time.time()
    label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']
    
    # Initialize video processing
    detect = get_face_areas.VideoCamera(path_video=path, conf=args.conf_d)
    dict_face_areas, total_frame = detect.get_frame()
    
    if not dict_face_areas:
        print(f"No valid frames found in video: {path}")
        return

    name_frames = list(dict_face_areas.keys())
    face_areas = list(dict_face_areas.values())

    try:
        # Load models
        EE_model = load_model_with_custom_names(args.path_FE_model)
        if EE_model is None:
            raise RuntimeError("Failed to load EE model")
            
        LSTM_model = load_model_with_custom_names(args.path_LSTM_model)
        if LSTM_model is None:
            raise RuntimeError("Failed to load LSTM model")

        # Feature extraction and prediction
        face_areas_array = np.stack(face_areas)
        if len(face_areas_array.shape) != 4:
            print(f"Error: Invalid face areas shape {face_areas_array.shape}")
            return
            
        # Ensure face_areas_array has the correct shape (batch_size, 224, 224, 3)
        if face_areas_array.shape[1:] != (224, 224, 3):
            print(f"Error: Face areas have incorrect dimensions. Expected (N, 224, 224, 3), got {face_areas_array.shape}")
            return
            
        # Process features in batches if needed
        features = EE_model(face_areas_array)
        if isinstance(features, tf.Tensor):
            features = features.numpy()
            
        seq_paths, seq_features = sequences.sequences(name_frames, features)
        
        if not seq_features:
            print("No valid sequences found in the video")
            return
            
        seq_features_array = np.array(seq_features)
        pred = LSTM_model(seq_features_array).numpy()
        
        # Prepare predictions
        all_pred, all_path = [], []
        for id, c_p in enumerate(seq_paths):
            c_f = [str(i).zfill(6) for i in range(int(c_p[0]), int(c_p[-1]) + 1)]
            c_pr = [pred[id]] * len(c_f)
            all_pred.extend(c_pr)
            all_path.extend(c_f)
        
        m_f = [str(i).zfill(6) for i in range(int(all_path[-1]) + 1, total_frame + 1)]
        m_p = [all_pred[-1]] * len(m_f)
        
        df = pd.DataFrame(data=all_pred + m_p, columns=label_model)
        df['frame'] = all_path + m_f
        df = df[['frame'] + label_model]
        df = sequences.df_group(df, label_model)
        
        # Save results
        filename = os.path.basename(path).replace('.mp4', '.csv')
        df.to_csv(os.path.join(args.path_save, filename), index=False)
        end_time = time.time() - start_time
        
        mode = stats.mode(np.argmax(pred, axis=1))[0][0]
        print(f'Report saved in: {os.path.join(args.path_save, filename)}')
        print(f'Predicted emotion: {label_model[mode]}')
        print(f'Lead time: {np.round(end_time, 2)} s\n')
    except Exception as e:
        print(f"Error processing video {path}: {e}")

def pred_all_video():
    """Predict emotions for all videos in the directory."""
    path_all_videos = [f for f in os.listdir(args.path_video) if f.endswith(('.mp4', '.avi'))]
    if not path_all_videos:
        print("No video files found in the specified directory.")
        return
    
    for id, cr_path in enumerate(path_all_videos):
        print(f'Processing video {id + 1}/{len(path_all_videos)}: {cr_path}')
        pred_one_video(os.path.join(args.path_video, cr_path))

if __name__ == "__main__":
    pred_all_video()
