# Multi-Modal Temporal Emotion Recognition System 🎭

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📝 Description

A sophisticated real-time emotion recognition system that combines CNN and LSTM architectures to perform temporal emotion analysis from video inputs. The system leverages deep learning to detect and analyze facial expressions, providing both real-time and batch processing capabilities.

## ✨ Features

- 🎥 Real-time webcam-based emotion detection
- 🧠 Hybrid CNN-LSTM architecture for temporal analysis
- 👤 Robust face detection and feature extraction
- 📊 Support for multiple emotion classes
- ⚡ High-performance inference
- 📈 Temporal sequence analysis
- 🔄 Multi-framework support (TensorFlow & PyTorch)

## 🛠️ Technologies Used

- Python 3.8+
- TensorFlow 2.17.0
- PyTorch
- OpenCV
- Mediapipe
- Batch-face
- Scikit-learn
- JupyterLab

## 📋 Requirements

```bash
jupyterlab==4.2.4
tensorflow==2.17.0
scikit-learn==1.5.1
batch-face==1.5.0
mediapipe==0.10.14
seaborn==0.13.2
tqdm==4.66.4
torch
torchvision
torchaudio
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Tameem2003/Emotion-Recognition.git
cd Emotion-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models (if available) and place them in the `models/` directory.

## 💻 Usage

### Running Video Analysis
```bash
python run.py --path_video 'video/' --path_save 'report/' --conf_d 0.7
```

### Available Notebooks
- `check_backbone_models_by_webcam.ipynb`: Test backbone models using webcam
- `check_temporal_models_by_video_multi.ipynb`: Multiple video analysis
- `check_temporal_models_by_webcam.ipynb`: Real-time webcam analysis
- `test_LSTM_RAVDESS.ipynb`: LSTM model testing on RAVDESS dataset

## 📊 Supported Emotions

- Neutral
- Happiness
- Sadness
- Surprise
- Fear
- Disgust
- Anger

## 🏗️ Project Structure

```
.
├── models/                  # Pre-trained model weights
├── functions/              # Core functionality modules
├── video/                  # Input video directory
├── report/                 # Analysis reports
└── notebooks/             # Jupyter notebooks for testing
```

## 📈 Performance

The system achieves real-time performance with:
- Fast face detection and tracking
- Efficient feature extraction
- Optimized temporal analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

[Your Name/Team Names]

## 🙏 Acknowledgments

- AffectNet Dataset
- RAVDESS Dataset
- [Other acknowledgments]

---
⭐ Star this repository if you find it helpful!
