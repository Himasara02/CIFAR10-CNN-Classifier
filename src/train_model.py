"""
CIFAR-10 Training Script
=======================

This script handles the complete training pipeline for the CIFAR-10 CNN model.
It includes data loading, preprocessing, model training, and performance visualization.

Features:
- Automatic CIFAR-10 dataset loading and preprocessing
- Model training with validation monitoring
- Training history visualization
- Model saving for future use

Author: Your Name
Date: October 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))
from model import CIFAR10CNN
from utils import load_and_preprocess_data, plot_training_history, visualize_sample_images


def train_cifar10_model(epochs=10, batch_size=32, save_model=True):
    """
    Complete training pipeline for CIFAR-10 CNN model.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        save_model (bool): Whether to save the trained model
    
    Returns:
        tuple: (trained_model, training_history)
    """
    print("🚀 Starting CIFAR-10 CNN Training Pipeline...")
    print("="*50)
    
    # 1. Load and preprocess data
    print("📊 Loading and preprocessing CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    print(f"✅ Dataset loaded:")
    print(f"   - Training samples: {train_images.shape[0]}")
    print(f"   - Test samples: {test_images.shape[0]}")
    print(f"   - Image shape: {train_images.shape[1:]}")
    
    # 2. Visualize sample images
    print("\n🖼️  Visualizing sample images...")
    visualize_sample_images(train_images, train_labels)
    plt.savefig('../results/sample_images.png', dpi=300, bbox_inches='tight')
    print("   - Sample images saved to results/sample_images.png")
    
    # 3. Create and compile model
    print("\n🧠 Building CNN model...")
    cnn = CIFAR10CNN()
    model = cnn.build_model()
    cnn.compile_model()
    
    print("📋 Model Architecture:")
    cnn.get_model_summary()
    
    # 4. Train the model
    print(f"\n🏋️  Training model for {epochs} epochs...")
    print("="*50)
    
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_images, test_labels),
        verbose=1
    )
    
    # 5. Evaluate model performance
    print("\n📈 Evaluating model performance...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    
    print(f"🎯 Final Results:")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 6. Plot training history
    print("\n📊 Generating training plots...")
    plot_training_history(history)
    plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
    print("   - Training history saved to results/training_history.png")
    
    # 7. Save model
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'../models/cifar10_cnn_{timestamp}.h5'
        cnn.save_model(model_path)
        
        # Also save as latest model
        latest_model_path = '../models/cifar10_cnn_latest.h5'
        cnn.save_model(latest_model_path)
    
    print("\n🎉 Training completed successfully!")
    print("="*50)
    
    return model, history


def main():
    """Main function to run the training pipeline."""
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Train the model
    model, history = train_cifar10_model(
        epochs=10,
        batch_size=32,
        save_model=True
    )
    
    # Print final statistics
    print(f"\n📊 Training Summary:")
    print(f"   - Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"   - Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   - Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()