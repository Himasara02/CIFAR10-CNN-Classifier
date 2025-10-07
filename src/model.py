"""
CIFAR-10 CNN Model Architecture
==============================

This module defines the CNN architecture for CIFAR-10 image classification.
The model consists of feature extraction layers (Conv2D + MaxPooling) followed
by classification layers (Dense).

Architecture:
- Conv2D(64, 3x3) -> MaxPooling2D(2x2)
- Conv2D(128, 3x3) -> MaxPooling2D(2x2)
- Flatten -> Dense(120) -> Dense(84) -> Dense(10)

Author: Your Name
Date: October 2025
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class CIFAR10CNN:
    """
    CIFAR-10 Convolutional Neural Network model class.
    
    This class encapsulates the CNN architecture optimized for CIFAR-10
    classification with 10 output classes.
    """
    
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        """
        Initialize the CNN model.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def build_model(self):
        """
        Build the CNN architecture.
        
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        self.model = models.Sequential([
            # Feature Extraction Layers
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            
            # Classification Layers
            layers.Flatten(),
            layers.Dense(120, activation='relu'),
            layers.Dense(84, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        """
        Compile the model with specified optimizer and loss function.
        
        Args:
            optimizer (str): Optimizer for training
            loss (str): Loss function
            metrics (list): List of metrics to track
        """
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def get_model_summary(self):
        """
        Print model architecture summary.
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train the model first.")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


def create_model():
    """
    Factory function to create and return a CIFAR-10 CNN model.
    
    Returns:
        CIFAR10CNN: Initialized CNN model instance
    """
    cnn = CIFAR10CNN()
    cnn.build_model()
    cnn.compile_model()
    return cnn


if __name__ == "__main__":
    # Test the model creation
    print("🚀 Creating CIFAR-10 CNN Model...")
    model = create_model()
    print("\n📊 Model Architecture:")
    model.get_model_summary()
    print("\n✅ Model created successfully!")