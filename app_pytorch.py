"""
Flask web application với PyTorch model
Thay thế hoàn toàn TensorFlow/Keras
"""
import os
import sys
import logging
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Constants
MODEL_PATH = 'models/cat_dog_model_pytorch.pth'
MODEL_PATH_FALLBACK = 'models/cat_dog_model_pytorch_full.pth'
IMG_SIZE = 150
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch Model class (phải giống với train_model_pytorch.py)
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 9 * 9, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x)
        x = self.dropout1(torch.relu(self.bn5(self.fc1(x))))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = None


def load_model():
    """Load PyTorch model"""
    global model
    
    if model is not None:
        return model
    
    model_paths = [
        ('models/cat_dog_model_pytorch_full.pth', 'full'),
        ('models/cat_dog_model_pytorch.pth', 'state_dict'),
        ('models/cat_dog_model_pytorch_final.pth', 'state_dict')
    ]
    
    for path, load_type in model_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Đang tải mô hình từ {path}...")
                model = CatDogCNN().to(DEVICE)
                
                if load_type == 'full':
                    model = torch.load(path, map_location=DEVICE)
                else:
                    model.load_state_dict(torch.load(path, map_location=DEVICE))
                
                model.eval()  # Set to evaluation mode
                logger.info(f"Mô hình đã được tải thành công từ {path}!")
                logger.info(f"Device: {DEVICE}")
                return model
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình từ {path}: {str(e)}")
                continue
    
    logger.warning("Không tìm thấy mô hình PyTorch. Vui lòng chạy train_model_pytorch.py để huấn luyện.")
    return None


def allowed_file(filename):
    """Kiểm tra extension của file"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    """Tiền xử lý ảnh cho PyTorch model"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(DEVICE)


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
    
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Mô hình chưa được huấn luyện. Vui lòng chạy train_model_pytorch.py để huấn luyện mô hình trước.'
            }), 503
    
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
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Định dạng file không được hỗ trợ. Các định dạng được hỗ trợ: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        file_content = file.read()
        
        if len(file_content) > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File quá lớn. Kích thước tối đa: {MAX_FILE_SIZE / 1024 / 1024}MB'
            }), 400
        
        if len(file_content) == 0:
            return jsonify({
                'success': False,
                'error': 'File rỗng'
            }), 400
        
        # Đọc ảnh
        try:
            image = Image.open(io.BytesIO(file_content))
        except Exception as e:
            logger.error(f"Lỗi đọc ảnh: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Không thể đọc file ảnh: {str(e)}'
            }), 400
        
        # Tiền xử lý
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
            with torch.no_grad():
                output = model(processed_image)
                probability = float(output.item())
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
            'device': str(DEVICE),
            'model_type': 'PyTorch'
        })
    except Exception as e:
        logger.error(f"Lỗi health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File quá lớn. Kích thước tối đa: 16MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint không tồn tại'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    logger.error(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': 'Lỗi server nội bộ'
    }), 500


if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Khởi động ứng dụng Cat/Dog Classifier với PyTorch")
    logger.info("=" * 50)
    
    if model is not None:
        logger.info("✓ Mô hình đã sẵn sàng")
    else:
        logger.warning("⚠ Mô hình chưa được tải")
        logger.warning("⚠ Ứng dụng sẽ chạy nhưng không thể dự đoán cho đến khi mô hình được tải")
    
    logger.info(f"Device: {DEVICE}")
    logger.info("Truy cập http://localhost:5000 để sử dụng")
    logger.info("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Lỗi khởi động server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

