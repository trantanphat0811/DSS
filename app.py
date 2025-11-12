"""
Flask web application để phân loại Cat và Dog từ ảnh upload
Nâng cấp với error handling và validation tốt hơn
"""
import os
import sys
import logging
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify
try:
    import tensorflow as tf
    keras = tf.keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        import keras
from PIL import Image
import io

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Constants
MODEL_PATH = 'models/cat_dog_model.h5'
MODEL_PATH_FALLBACK = 'models/cat_dog_model_final.h5'
IMG_SIZE = 150
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Global model variable
model = None


def load_model():
    """Load mô hình với error handling và fallback"""
    global model
    
    if model is not None:
        return model
    
    model_paths = [MODEL_PATH, MODEL_PATH_FALLBACK]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Đang tải mô hình từ {path}...")
                model = keras.models.load_model(path, compile=False)
                # Compile lại model để đảm bảo tương thích
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
                    metrics=['accuracy']
                )
                logger.info(f"Mô hình đã được tải thành công từ {path}!")
                return model
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình từ {path}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
    
    logger.warning("Không tìm thấy mô hình nào. Vui lòng chạy train_model.py để huấn luyện mô hình.")
    return None


def allowed_file(filename):
    """Kiểm tra extension của file có hợp lệ không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(image):
    """Validate và kiểm tra ảnh có hợp lệ không"""
    try:
        # Kiểm tra mode
        if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
            image = image.convert('RGB')
        
        # Kiểm tra kích thước tối thiểu
        if image.size[0] < 32 or image.size[1] < 32:
            return False, "Ảnh quá nhỏ (tối thiểu 32x32 pixels)"
        
        # Kiểm tra kích thước tối đa
        if image.size[0] > 10000 or image.size[1] > 10000:
            return False, "Ảnh quá lớn (tối đa 10000x10000 pixels)"
        
        # Verify ảnh có thể load được
        image.verify()
        
        return True, None
    except Exception as e:
        logger.error(f"Lỗi validate ảnh: {str(e)}")
        return False, f"Ảnh không hợp lệ: {str(e)}"


def preprocess_image(image):
    """Tiền xử lý ảnh trước khi đưa vào mô hình"""
    try:
        # Đảm bảo ảnh ở chế độ RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ảnh về kích thước yêu cầu (sử dụng LANCZOS để chất lượng tốt hơn)
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Chuyển thành numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Kiểm tra shape
        if img_array.shape != (IMG_SIZE, IMG_SIZE, 3):
            raise ValueError(f"Shape không đúng: {img_array.shape}, mong đợi ({IMG_SIZE}, {IMG_SIZE}, 3)")
        
        # Normalize về [0, 1]
        img_array = img_array / 255.0
        
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Lỗi preprocess ảnh: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# Load model khi khởi động
model = load_model()


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để nhận ảnh và trả về kết quả dự đoán"""
    global model
    
    # Kiểm tra model
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Mô hình chưa được huấn luyện. Vui lòng chạy train_model.py để huấn luyện mô hình trước.'
            }), 503
    
    # Kiểm tra file upload
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Không có file ảnh được upload'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Chưa chọn file'
        }), 400
    
    # Kiểm tra extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Định dạng file không được hỗ trợ. Các định dạng được hỗ trợ: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Đọc file
        file_content = file.read()
        
        # Kiểm tra kích thước file
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File quá lớn. Kích thước tối đa: {MAX_FILE_SIZE / 1024 / 1024}MB'
            }), 400
        
        # Kiểm tra file rỗng
        if len(file_content) == 0:
            return jsonify({
                'success': False,
                'error': 'File rỗng'
            }), 400
        
        # Đọc ảnh từ file
        try:
            image = Image.open(io.BytesIO(file_content))
        except Exception as e:
            logger.error(f"Lỗi đọc ảnh: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Không thể đọc file ảnh: {str(e)}'
            }), 400
        
        # Validate ảnh
        is_valid, error_msg = validate_image(image.copy())
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Reset file pointer và load lại ảnh (vì verify() có thể làm hỏng ảnh)
        image = Image.open(io.BytesIO(file_content))
        
        # Tiền xử lý ảnh
        try:
            processed_image = preprocess_image(image)
        except Exception as e:
            logger.error(f"Lỗi preprocess: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Lỗi xử lý ảnh: {str(e)}'
            }), 500
        
        # Dự đoán
        try:
            prediction = model.predict(processed_image, verbose=0)
            probability = float(prediction[0][0])
            
            # Đảm bảo probability trong khoảng [0, 1]
            probability = np.clip(probability, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Lỗi prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Lỗi khi dự đoán: {str(e)}'
            }), 500
        
        # Xác định kết quả
        if probability > 0.5:
            result = 'Dog'
            confidence = probability * 100
        else:
            result = 'Cat'
            confidence = (1 - probability) * 100
        
        logger.info(f"Prediction: {result} với confidence {confidence:.2f}%")
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'probability': round(probability, 4)
        })
    
    except Exception as e:
        logger.error(f"Lỗi không xác định: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Kiểm tra trạng thái của ứng dụng"""
    global model
    
    try:
        model_status = model is not None
        if model is None:
            model = load_model()
            model_status = model is not None
        
        return jsonify({
            'status': 'healthy' if model_status else 'model_missing',
            'model_loaded': model_status,
            'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_FALLBACK if os.path.exists(MODEL_PATH_FALLBACK) else None
        })
    except Exception as e:
        logger.error(f"Lỗi health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Xử lý lỗi file quá lớn"""
    return jsonify({
        'success': False,
        'error': 'File quá lớn. Kích thước tối đa: 16MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Xử lý lỗi 404"""
    return jsonify({
        'success': False,
        'error': 'Endpoint không tồn tại'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Xử lý lỗi 500"""
    logger.error(f"Internal server error: {str(error)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': 'Lỗi server nội bộ'
    }), 500


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Khởi động ứng dụng Cat/Dog Classifier")
    logger.info("=" * 50)
    
    if model is not None:
        logger.info("✓ Mô hình đã sẵn sàng")
    else:
        logger.warning("⚠ Mô hình chưa được tải")
        logger.warning("⚠ Ứng dụng sẽ chạy nhưng không thể dự đoán cho đến khi mô hình được tải")
    
    logger.info("Truy cập http://localhost:5050 để sử dụng")
    logger.info("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5050)
    except Exception as e:
        logger.error(f"Lỗi khởi động server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
