"""
CIFAR-10 CNN Project Package
===========================

This package contains modules for training and using a Convolutional Neural Network
to classify CIFAR-10 images into 10 categories.

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .model import CIFAR10CNN
from .utils import (
    load_and_preprocess_data,
    get_class_names,
    visualize_prediction,
    plot_training_history,
    evaluate_model_performance
)

__all__ = [
    "CIFAR10CNN",
    "load_and_preprocess_data", 
    "get_class_names",
    "visualize_prediction",
    "plot_training_history",
    "evaluate_model_performance"
]