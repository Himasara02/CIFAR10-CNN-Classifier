"""
CIFAR-10 CNN Prediction Module
=============================

This module provides functions for making predictions with the trained CIFAR-10 CNN model,
including single image prediction, batch prediction, and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
from PIL import Image
from utils import get_class_names, denormalize_image, visualize_prediction


class CIFAR10Predictor:
    """A class for making predictions with the trained CIFAR-10 CNN model."""
    
    def __init__(self, model_path="models/cifar10_cnn_model.h5"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = get_class_names()
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"⚠️  Model not found at {model_path}")
            print("Please train the model first using train_model.py")
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction.
        
        Args:
            image (np.array): Input image of shape (32, 32, 3)
            
        Returns:
            np.array: Preprocessed image ready for prediction
        """
        # Ensure image is in correct format
        if len(image.shape) == 3 and image.shape[-1] == 3:
            # Normalize to [-1, 1] range
            processed_image = image.astype('float32') / 127.5 - 1
            # Add batch dimension
            processed_image = np.expand_dims(processed_image, axis=0)
            return processed_image
        else:
            raise ValueError("Image must be RGB with shape (32, 32, 3)")
    
    def predict_single(self, image, visualize=True):
        """
        Make a prediction on a single image.
        
        Args:
            image (np.array): Input image of shape (32, 32, 3)
            visualize (bool): Whether to show visualization
            
        Returns:
            dict: Prediction results including probabilities and predicted class
        """
        if self.model is None:
            print("❌ No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction
            probabilities = self.model.predict(processed_image, verbose=0)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Create results dictionary
            results = {
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'probabilities': probabilities,
                'all_predictions': {self.class_names[i]: prob for i, prob in enumerate(probabilities)}
            }
            
            # Print results
            print(f"🔍 Prediction Results:")
            print(f"   Predicted Class: {predicted_class}")
            print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # Visualize if requested
            if visualize:
                visualize_prediction(image, probabilities)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None
    
    def predict_batch(self, images, return_probabilities=False):
        """
        Make predictions on a batch of images.
        
        Args:
            images (np.array): Batch of images of shape (N, 32, 32, 3)
            return_probabilities (bool): Whether to return full probability distributions
            
        Returns:
            dict: Batch prediction results
        """
        if self.model is None:
            print("❌ No model loaded. Cannot make predictions.")
            return None
        
        try:
            # Preprocess images
            processed_images = images.astype('float32') / 127.5 - 1
            
            # Make predictions
            probabilities = self.model.predict(processed_images, verbose=0)
            predicted_classes_idx = np.argmax(probabilities, axis=1)
            predicted_classes = [self.class_names[idx] for idx in predicted_classes_idx]
            confidences = np.max(probabilities, axis=1)
            
            results = {
                'predicted_classes': predicted_classes,
                'predicted_classes_idx': predicted_classes_idx,
                'confidences': confidences,
                'mean_confidence': np.mean(confidences)
            }
            
            if return_probabilities:
                results['probabilities'] = probabilities
            
            print(f"✅ Batch prediction completed for {len(images)} images")
            print(f"   Average confidence: {np.mean(confidences):.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during batch prediction: {str(e)}")
            return None
    
    def predict_test_samples(self, num_samples=5):
        """
        Predict on random test samples from CIFAR-10 dataset.
        
        Args:
            num_samples (int): Number of test samples to predict
        """
        if self.model is None:
            print("❌ No model loaded. Cannot make predictions.")
            return
        
        # Load test data
        print("🔄 Loading CIFAR-10 test data...")
        (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        
        # Select random samples
        random_indices = np.random.choice(len(test_images), num_samples, replace=False)
        
        print(f"🧪 Making predictions on {num_samples} random test samples:")
        print("=" * 60)
        
        correct_predictions = 0
        
        for i, idx in enumerate(random_indices):
            image = test_images[idx]
            true_label = test_labels[idx][0]
            true_class = self.class_names[true_label]
            
            print(f"\n📊 Sample {i+1}/{num_samples} (Index: {idx})")
            print(f"   True Label: {true_class}")
            
            # Make prediction
            result = self.predict_single(image, visualize=False)
            
            if result:
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                
                is_correct = predicted_class == true_class
                if is_correct:
                    correct_predictions += 1
                    print(f"   ✅ Correct! Confidence: {confidence:.4f}")
                else:
                    print(f"   ❌ Incorrect. Predicted: {predicted_class}, Confidence: {confidence:.4f}")
        
        accuracy = correct_predictions / num_samples
        print(f"\n📈 Sample Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Correct Predictions: {correct_predictions}/{num_samples}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN Prediction Tool")
    parser.add_argument("--model", type=str, default="models/cifar10_cnn_model.h5",
                        help="Path to the trained model file")
    parser.add_argument("--test-samples", type=int, default=5,
                        help="Number of test samples to predict")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a custom image file for prediction")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CIFAR10Predictor(model_path=args.model)
    
    if args.image:
        # Predict on custom image
        try:
            # Load and resize image
            img = Image.open(args.image)
            img = img.resize((32, 32))
            img_array = np.array(img)
            
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            print(f"🖼️  Predicting custom image: {args.image}")
            predictor.predict_single(img_array, visualize=True)
            
        except Exception as e:
            print(f"❌ Error loading custom image: {str(e)}")
    else:
        # Predict on test samples
        predictor.predict_test_samples(num_samples=args.test_samples)


if __name__ == "__main__":
    main()
