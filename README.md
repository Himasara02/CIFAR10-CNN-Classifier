# 🖼️ CIFAR-10 Image Classification with CNN

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Convolutional Neural Network (CNN)** implementation in Python using TensorFlow for classifying CIFAR-10 images into 10 categories. Achieved **90% training accuracy** and **70% validation accuracy** with comprehensive preprocessing, normalization, and real-time prediction visualization.

## 🎯 Project Overview

This project demonstrates end-to-end deep learning workflow for image classification:
- **Dataset**: CIFAR-10 (60,000 32x32 color images in 10 classes)
- **Architecture**: Custom CNN with feature extraction and classification layers
- **Performance**: 90% training accuracy, 70% validation accuracy
- **Visualization**: Real-time prediction analysis with probability distributions

## 🏗️ CNN Architecture

```
Input (32x32x3)
    ↓
Conv2D(64, 3x3, ReLU)
    ↓
MaxPooling2D(2x2)
    ↓
Conv2D(128, 3x3, ReLU)
    ↓
MaxPooling2D(2x2)
    ↓
Flatten()
    ↓
Dense(120, ReLU)
    ↓
Dense(84, ReLU)
    ↓
Dense(10, Softmax)
```

## 📊 Classes

The model classifies images into these 10 categories:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚛 Truck

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/CIFAR10-CNN-Classifier.git
cd CIFAR10-CNN-Classifier
pip install -r requirements.txt
```

### Usage

#### 1. Train the Model
```python
python src/train_model.py
```

#### 2. Make Predictions
```python
python src/predict.py --image_path path/to/your/image.jpg
```

#### 3. Run Jupyter Notebook (Interactive)
```bash
jupyter notebook notebooks/cifar10_classifier.ipynb
```

## 📁 Project Structure

```
CIFAR10-CNN-Classifier/
├── 📂 src/
│   ├── 🐍 model.py          # CNN architecture
│   ├── 🐍 train_model.py    # Training script
│   ├── 🐍 predict.py        # Prediction script
│   └── 🐍 utils.py          # Utility functions
├── 📂 notebooks/
│   └── 📓 cifar10_classifier.ipynb
├── 📂 models/
│   └── 💾 trained_model.h5
├── 📂 data/
│   └── 📊 (CIFAR-10 auto-downloaded)
├── 📂 results/
│   └── 📈 training_plots.png
├── 📋 requirements.txt
├── 📖 README.md
└── 🚫 .gitignore
```

## 🔧 Key Features

### 🎨 Preprocessing & Normalization
- Pixel normalization: [0, 255] → [-1, 1]
- Automatic data loading and splitting
- Image denormalization for visualization

### 🧠 CNN Architecture
- **Feature Extraction**: Convolutional + Pooling layers
- **Classification**: Fully connected dense layers
- **Activation**: ReLU for hidden layers, Softmax for output
- **Optimization**: Adam optimizer with sparse categorical crossentropy

### 📊 Visualization
- Sample image gallery with class labels
- Real-time prediction probability bars
- Training history plots (accuracy/loss)
- Model architecture summary

### ⚡ Performance
- **Training Accuracy**: ~90%
- **Validation Accuracy**: ~70%
- **Training Time**: ~10 epochs
- **Model Size**: Lightweight for deployment

## 📈 Results

### Training Performance
- **Final Training Accuracy**: 90.23%
- **Final Validation Accuracy**: 70.15%
- **Model Convergence**: Stable after 8-10 epochs
- **Overfitting Control**: Validation monitoring

### Sample Predictions
The model provides confidence scores for all 10 classes, enabling detailed analysis of prediction uncertainty and class confusion patterns.

## 🛠️ Technical Implementation

### Libraries Used
- **TensorFlow 2.x**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Jupyter**: Interactive development

### Model Training
```python
# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)
```

## 📚 Learning Outcomes

This project demonstrates:
- ✅ CNN architecture design for image classification
- ✅ Data preprocessing and normalization techniques
- ✅ Model training with validation monitoring
- ✅ Real-time prediction visualization
- ✅ Performance evaluation and metrics analysis

## 🔄 Future Enhancements

- [ ] Data augmentation for improved generalization
- [ ] Transfer learning with pre-trained models
- [ ] Hyperparameter optimization
- [ ] Model deployment with Flask/FastAPI
- [ ] Real-time webcam classification

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

 
