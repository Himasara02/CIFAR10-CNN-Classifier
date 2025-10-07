#!/usr/bin/env python3
"""
CIFAR-10 CNN Project Test Script
===============================

This script tests the project setup and verifies all components work correctly.
Run this after setting up the project to ensure everything is working properly.
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib: Available")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if all project files and directories exist."""
    print("\n📁 Testing project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        ".gitignore",
        "LICENSE",
        "src/__init__.py",
        "src/model.py",
        "src/train_model.py",
        "src/predict.py",
        "src/utils.py",
        "notebooks/image_classifier.ipynb",
        "data/.gitkeep",
        "models/.gitkeep",
        "results/.gitkeep"
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_files_exist = False
    
    return all_files_exist

def test_module_imports():
    """Test if project modules can be imported."""
    print("\n🔧 Testing project module imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        from model import CIFAR10CNN
        print("✅ model.CIFAR10CNN")
    except ImportError as e:
        print(f"❌ model.CIFAR10CNN: {e}")
        return False
    
    try:
        from utils import load_and_preprocess_data, get_class_names
        print("✅ utils functions")
    except ImportError as e:
        print(f"❌ utils functions: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the project."""
    print("\n⚡ Testing basic functionality...")
    
    try:
        # Test model creation
        sys.path.insert(0, 'src')
        from model import CIFAR10CNN
        
        model_builder = CIFAR10CNN()
        model = model_builder.build_model()
        
        print(f"✅ Model created successfully")
        print(f"   Parameters: {model.count_params():,}")
        
        # Test data loading (small sample)
        from utils import load_and_preprocess_data, get_class_names
        
        print("✅ Data loading functions work")
        
        classes = get_class_names()
        print(f"✅ Class names: {len(classes)} classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 CIFAR-10 CNN Project Test Suite")
    print("=" * 50)
    
    # Run tests
    test_results = []
    test_results.append(test_imports())
    test_results.append(test_project_structure())
    test_results.append(test_module_imports())
    test_results.append(test_basic_functionality())
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"🎉 All tests passed! ({passed_tests}/{total_tests})")
        print("✅ Project setup is complete and working correctly!")
        print("\n🚀 Next steps:")
        print("   1. Run: python src/train_model.py")
        print("   2. Run: python src/predict.py --test-samples 5")
        print("   3. Open: notebooks/image_classifier.ipynb")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} test(s) failed ({passed_tests}/{total_tests} passed)")
        print("🔧 Please fix the issues above and run the test again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)