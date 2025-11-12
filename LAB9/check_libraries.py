#!/usr/bin/env python3
"""
Script kiểm tra thư viện có sẵn trên máy
"""
import sys

print("=" * 60)
print("Kiểm tra thư viện có sẵn")
print("=" * 60)

libraries = {
    'Keras': None,
    'TensorFlow': None,
    'PyTorch': None,
    'scikit-learn': None,
    'Pillow': None,
    'NumPy': None,
    'Matplotlib': None,
    'Flask': None,
}

# Kiểm tra từng thư viện
for lib_name in libraries.keys():
    try:
        if lib_name == 'Keras':
            try:
                import keras
                libraries[lib_name] = keras.__version__ if hasattr(keras, '__version__') else 'OK'
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                try:
                    import tensorflow as tf
                    libraries[lib_name] = f"TensorFlow Keras {tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'OK'}"
                    print(f"✓ {lib_name}: {libraries[lib_name]}")
                except:
                    print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'TensorFlow':
            try:
                import tensorflow as tf
                libraries[lib_name] = tf.__version__ if hasattr(tf, '__version__') else 'OK'
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'PyTorch':
            try:
                import torch
                libraries[lib_name] = torch.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'scikit-learn':
            try:
                import sklearn
                libraries[lib_name] = sklearn.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'Pillow':
            try:
                import PIL
                libraries[lib_name] = PIL.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'NumPy':
            try:
                import numpy as np
                libraries[lib_name] = np.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'Matplotlib':
            try:
                import matplotlib
                libraries[lib_name] = matplotlib.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
        
        elif lib_name == 'Flask':
            try:
                import flask
                libraries[lib_name] = flask.__version__
                print(f"✓ {lib_name}: {libraries[lib_name]}")
            except:
                print(f"✗ {lib_name}: Not found")
    
    except Exception as e:
        print(f"✗ {lib_name}: Error - {str(e)}")

print("=" * 60)

# Đề xuất
print("\n📋 Đề xuất:")
if libraries['Keras'] or libraries['TensorFlow']:
    if libraries['TensorFlow']:
        print("  → Sử dụng: train_model.py (với TensorFlow/Keras)")
    else:
        print("  → Sử dụng: train_model.py hoặc train_model_keras_standalone.py")
else:
    print("  → Cần cài đặt: pip install keras hoặc pip install tensorflow")

if not libraries['scikit-learn']:
    print("  ⚠ Cần cài đặt scikit-learn cho train_model_keras_standalone.py")
    print("     pip install scikit-learn")

print("=" * 60)

