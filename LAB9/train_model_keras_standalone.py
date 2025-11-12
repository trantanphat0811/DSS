"""
Script huấn luyện mô hình CNN sử dụng Keras standalone
Không phụ thuộc vào TensorFlow ImageDataGenerator
"""
import os
import sys
import logging
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import Keras - thử nhiều cách
try:
    # Thử Keras standalone trước
    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    logger.info("Đã sử dụng Keras standalone")
except ImportError:
    try:
        # Thử TensorFlow Keras
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        logger.info("Đã sử dụng TensorFlow Keras")
    except ImportError:
        logger.error("Không tìm thấy Keras hoặc TensorFlow!")
        sys.exit(1)

# Đường dẫn đến dữ liệu
DATA_DIR = 'data/kagglecatsanddogs_3367a/PetImages'
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10


def load_images_from_directory(directory, label, max_images=None):
    """Load ảnh từ thư mục và gán label"""
    images = []
    labels = []
    image_paths = []
    
    # Lấy danh sách file ảnh
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))
        image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    logger.info(f"Đang load {len(image_paths)} ảnh từ {directory}...")
    
    for img_path in image_paths:
        try:
            # Đọc và resize ảnh
            img = Image.open(img_path)
            
            # Chuyển sang RGB nếu cần
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            
            # Chuyển thành array và normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Kiểm tra shape
            if img_array.shape == (IMG_SIZE, IMG_SIZE, 3):
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            logger.warning(f"Không thể load ảnh {img_path}: {str(e)}")
            continue
    
    return images, labels


def augment_image(img_array):
    """Data augmentation đơn giản"""
    from scipy.ndimage import rotate, zoom, shift
    import random
    
    augmented = []
    
    # Original
    augmented.append(img_array)
    
    # Horizontal flip
    augmented.append(np.fliplr(img_array))
    
    # Rotation
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        rotated = rotate(img_array, angle, axes=(0, 1), reshape=False, order=1, mode='reflect')
        rotated = np.clip(rotated, 0, 1)
        augmented.append(rotated)
    
    return augmented


def prepare_data():
    """Chuẩn bị dữ liệu từ thư mục"""
    cat_dir = os.path.join(DATA_DIR, 'Cat')
    dog_dir = os.path.join(DATA_DIR, 'Dog')
    
    # Load ảnh
    logger.info("Đang load ảnh Cat...")
    cat_images, cat_labels = load_images_from_directory(cat_dir, 0)
    
    logger.info("Đang load ảnh Dog...")
    dog_images, dog_labels = load_images_from_directory(dog_dir, 1)
    
    # Kết hợp
    all_images = cat_images + dog_images
    all_labels = cat_labels + dog_labels
    
    logger.info(f"Tổng số ảnh: {len(all_images)}")
    logger.info(f"  - Cat: {len(cat_images)}")
    logger.info(f"  - Dog: {len(dog_images)}")
    
    # Chuyển thành numpy arrays
    X = np.array(all_images)
    y = np.array(all_labels)
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


def create_model():
    """Tạo mô hình CNN"""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 4
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Classification head
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    
    return model


def train():
    """Huấn luyện mô hình"""
    try:
        # Kiểm tra dữ liệu
        if not os.path.exists(DATA_DIR):
            logger.error(f"Thư mục dữ liệu không tồn tại: {DATA_DIR}")
            return None
        
        # Chuẩn bị dữ liệu
        logger.info("Đang chuẩn bị dữ liệu...")
        X_train, X_val, y_train, y_val = prepare_data()
        
        # Tạo mô hình
        logger.info("Đang tạo mô hình...")
        model = create_model()
        model.summary()
        
        # Tạo thư mục models
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'models/cat_dog_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Huấn luyện
        logger.info("Bắt đầu huấn luyện...")
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        # Lưu mô hình cuối cùng
        logger.info("Đang lưu mô hình cuối cùng...")
        model.save('models/cat_dog_model_final.h5')
        
        # Vẽ biểu đồ
        try:
            plot_training_history(history)
        except Exception as e:
            logger.warning(f"Không thể vẽ biểu đồ: {str(e)}")
        
        # Thông tin kết quả
        best_val_acc = max(history.history['val_accuracy'])
        logger.info("=" * 50)
        logger.info("Huấn luyện hoàn tất!")
        logger.info(f"Độ chính xác tốt nhất (validation): {best_val_acc:.4f}")
        logger.info("=" * 50)
        
        return model
        
    except KeyboardInterrupt:
        logger.warning("\nHuấn luyện bị gián đoạn")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def plot_training_history(history):
    """Vẽ biểu đồ quá trình huấn luyện"""
    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training accuracy', linewidth=2, markersize=6)
        plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy', linewidth=2, markersize=6)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Training loss', linewidth=2, markersize=6)
        plt.plot(epochs, val_loss, 'ro-', label='Validation loss', linewidth=2, markersize=6)
        plt.title('Model Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        logger.info("Đã lưu biểu đồ vào models/training_history.png")
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ: {str(e)}")


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Bắt đầu huấn luyện mô hình Cat/Dog Classifier")
    logger.info("Sử dụng Keras standalone với xử lý dữ liệu thủ công")
    logger.info("=" * 50)
    
    try:
        model = train()
        if model is not None:
            logger.info("Huấn luyện thành công!")
        else:
            logger.error("Huấn luyện thất bại!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

