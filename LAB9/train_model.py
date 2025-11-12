"""
Script để huấn luyện mô hình CNN phân loại Cat và Dog
Nâng cấp với error handling và validation tốt hơn
Tương thích với Keras standalone và TensorFlow Keras
"""
import os
import sys
import logging
import traceback
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Cấu hình logging trước
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import Keras - thử nhiều cách để tương thích
keras_module = None
Sequential = None
Conv2D = None
MaxPooling2D = None
Flatten = None
Dense = None
Dropout = None
BatchNormalization = None
Adam = None
ModelCheckpoint = None
EarlyStopping = None
ReduceLROnPlateau = None
ImageDataGenerator = None
IMAGE_DATA_GENERATOR_AVAILABLE = False

try:
    # Ưu tiên Keras standalone
    import keras
    keras_module = keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    try:
        from keras.preprocessing.image import ImageDataGenerator
        IMAGE_DATA_GENERATOR_AVAILABLE = True
    except ImportError:
        pass
    logger.info("✓ Đã sử dụng Keras standalone")
except ImportError:
    try:
        # Thử TensorFlow Keras
        import tensorflow as tf
        keras_module = tf.keras
        Sequential = tf.keras.Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            IMAGE_DATA_GENERATOR_AVAILABLE = True
        except ImportError:
            pass
        logger.info("✓ Đã sử dụng TensorFlow Keras")
    except ImportError as e:
        logger.error(f"✗ Không tìm thấy Keras hoặc TensorFlow: {e}")
        logger.error("Vui lòng cài đặt: pip install keras hoặc pip install tensorflow")
        sys.exit(1)

# Nếu chưa import được, thử lại với cách khác
if Sequential is None:
    Sequential = keras_module.Sequential
if Conv2D is None:
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
if Adam is None:
    try:
        from keras.optimizers import Adam
    except:
        Adam = keras_module.optimizers.Adam
if ModelCheckpoint is None:
    try:
        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    except:
        ModelCheckpoint = keras_module.callbacks.ModelCheckpoint
        EarlyStopping = keras_module.callbacks.EarlyStopping
        ReduceLROnPlateau = keras_module.callbacks.ReduceLROnPlateau

if not IMAGE_DATA_GENERATOR_AVAILABLE:
    logger.warning("⚠ ImageDataGenerator không có sẵn, sẽ dùng xử lý dữ liệu thủ công nếu cần")

# Đường dẫn đến dữ liệu
DATA_DIR = 'data/kagglecatsanddogs_3367a/PetImages'
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10


def check_data_directory():
    """Kiểm tra thư mục dữ liệu có tồn tại và hợp lệ không"""
    if not os.path.exists(DATA_DIR):
        logger.error(f"Thư mục dữ liệu không tồn tại: {DATA_DIR}")
        return False
    
    cat_dir = os.path.join(DATA_DIR, 'Cat')
    dog_dir = os.path.join(DATA_DIR, 'Dog')
    
    if not os.path.exists(cat_dir):
        logger.error(f"Thư mục Cat không tồn tại: {cat_dir}")
        return False
    
    if not os.path.exists(dog_dir):
        logger.error(f"Thư mục Dog không tồn tại: {dog_dir}")
        return False
    
    # Đếm số file
    try:
        cat_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        dog_files = [f for f in os.listdir(dog_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        logger.info(f"Số ảnh Cat: {len(cat_files)}")
        logger.info(f"Số ảnh Dog: {len(dog_files)}")
        
        if len(cat_files) == 0 or len(dog_files) == 0:
            logger.error("Không có ảnh nào trong thư mục dữ liệu")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra dữ liệu: {str(e)}")
        return False


def create_model():
    """Tạo mô hình CNN với kiến trúc cải tiến"""
    try:
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
        
        logger.info("Mô hình đã được tạo thành công")
        return model
    except Exception as e:
        logger.error(f"Lỗi khi tạo mô hình: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def prepare_data():
    """Chuẩn bị dữ liệu với ImageDataGenerator và validation"""
    if not IMAGE_DATA_GENERATOR_AVAILABLE:
        logger.error("ImageDataGenerator không khả dụng. Vui lòng sử dụng train_model_keras_standalone.py")
        raise ImportError("ImageDataGenerator không khả dụng")
    
    try:
        # Data augmentation cho training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2,  # 20% dữ liệu dùng cho validation
            brightness_range=[0.8, 1.2]
        )
        
        # Chỉ rescale cho validation (không augmentation)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training generator
        logger.info("Đang tạo training generator...")
        train_generator = train_datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        logger.info("Đang tạo validation generator...")
        validation_generator = validation_datagen.flow_from_directory(
            DATA_DIR,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {validation_generator.samples}")
        logger.info(f"Classes: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị dữ liệu: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def train():
    """Huấn luyện mô hình với error handling"""
    try:
        # Kiểm tra dữ liệu
        logger.info("Đang kiểm tra thư mục dữ liệu...")
        if not check_data_directory():
            logger.error("Dữ liệu không hợp lệ. Vui lòng kiểm tra lại.")
            return None
        
        # Tạo mô hình
        logger.info("Đang tạo mô hình...")
        model = create_model()
        model.summary()
        
        # Chuẩn bị dữ liệu
        logger.info("Đang chuẩn bị dữ liệu...")
        train_gen, val_gen = prepare_data()
        
        # Tạo thư mục models nếu chưa có
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
        
        logger.info("Bắt đầu huấn luyện...")
        logger.info(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}")
        
        history = model.fit(
            train_gen,
            steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=max(1, val_gen.samples // BATCH_SIZE),
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
        best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
        
        logger.info("=" * 50)
        logger.info("Huấn luyện hoàn tất!")
        logger.info(f"Độ chính xác tốt nhất (validation): {best_val_acc:.4f} tại epoch {best_epoch}")
        logger.info(f"Độ chính xác cuối cùng (training): {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Độ chính xác cuối cùng (validation): {history.history['val_accuracy'][-1]:.4f}")
        logger.info("=" * 50)
        
        return model
        
    except KeyboardInterrupt:
        logger.warning("\nHuấn luyện bị gián đoạn bởi người dùng")
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
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Training accuracy', linewidth=2, markersize=6)
        plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy', linewidth=2, markersize=6)
        plt.title('Model Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Loss plot
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
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Bắt đầu huấn luyện mô hình Cat/Dog Classifier")
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
