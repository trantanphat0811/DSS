#!/usr/bin/env python3
"""
Script test TensorFlow/Keras installation
"""
import sys

print("=" * 60)
print("Kiểm tra TensorFlow/Keras Installation")
print("=" * 60)

# Test 1: Import TensorFlow
try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    try:
        version = tf.__version__
        print(f"  Version: {version}")
    except:
        print("  Version: (không thể lấy version)")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

# Test 2: Import Keras
try:
    from tensorflow import keras
    print("✓ Keras imported successfully")
    try:
        version = keras.__version__
        print(f"  Version: {version}")
    except:
        print("  Version: (không thể lấy version)")
except ImportError as e:
    print(f"✗ Keras import failed: {e}")
    sys.exit(1)

# Test 3: Import layers
try:
    from tensorflow.keras import layers
    print("✓ tensorflow.keras.layers imported successfully")
except ImportError as e:
    print(f"✗ tensorflow.keras.layers import failed: {e}")
    sys.exit(1)

# Test 4: Import ImageDataGenerator
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("✓ ImageDataGenerator imported successfully")
except ImportError as e:
    print(f"✗ ImageDataGenerator import failed: {e}")
    sys.exit(1)

# Test 5: Test model creation
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    print("✓ Model creation test: OK")
except Exception as e:
    print(f"✗ Model creation test failed: {e}")
    sys.exit(1)

# Test 6: Test ImageDataGenerator
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)
    print("✓ ImageDataGenerator creation test: OK")
except Exception as e:
    print(f"✗ ImageDataGenerator creation test failed: {e}")
    sys.exit(1)

print("=" * 60)
print("✓ Tất cả các test đều PASS!")
print("TensorFlow/Keras đã sẵn sàng để sử dụng")
print("=" * 60)

