# Hướng dẫn sử dụng - Cat & Dog Classifier

## 🔍 Kiểm tra thư viện có sẵn

Chạy script để kiểm tra:
```bash
python check_libraries.py
```

## 📦 Các phiên bản script training

### 1. `train_model_pytorch.py` ⭐ **MỚI - Khuyến nghị**
- **Yêu cầu**: PyTorch + torchvision
- **Ưu điểm**: 
  - Thay thế hoàn toàn TensorFlow/Keras
  - Tốc độ nhanh hơn, đặc biệt với GPU
  - Code rõ ràng, dễ hiểu
  - Hỗ trợ CUDA nếu có GPU
  - Data augmentation với torchvision
- **Sử dụng khi**: Có PyTorch hoặc muốn dùng thư viện khác TensorFlow

```bash
python train_model_pytorch.py
```

### 2. `train_model.py`
- **Yêu cầu**: TensorFlow/Keras với ImageDataGenerator
- **Ưu điểm**: 
  - Sử dụng ImageDataGenerator (hiệu quả với bộ nhớ lớn)
  - Data augmentation tự động
  - Tối ưu cho dataset lớn
- **Sử dụng khi**: Có TensorFlow hoặc Keras với ImageDataGenerator

```bash
python train_model.py
```

### 3. `train_model_keras_standalone.py` 
- **Yêu cầu**: Keras standalone hoặc TensorFlow/Keras + scikit-learn
- **Ưu điểm**:
  - Không cần ImageDataGenerator
  - Xử lý dữ liệu thủ công
  - Tương thích với nhiều môi trường
- **Sử dụng khi**: Không có ImageDataGenerator hoặc muốn kiểm soát dữ liệu tốt hơn

```bash
python train_model_keras_standalone.py
```

## 🛠️ Cài đặt thư viện

### Nếu thiếu PyTorch (Khuyến nghị):
```bash
pip install torch torchvision
```

### Nếu thiếu TensorFlow/Keras:
```bash
pip install tensorflow
# hoặc
pip install keras
```

### Nếu thiếu scikit-learn (cho version standalone):
```bash
pip install scikit-learn
```

### Cài đặt tất cả:
```bash
pip install -r requirements.txt
```

## ⚙️ Cấu hình

### Thay đổi tham số trong script:

```python
IMG_SIZE = 150        # Kích thước ảnh (150x150)
BATCH_SIZE = 32       # Batch size (giảm nếu thiếu RAM)
EPOCHS = 10           # Số epochs
```

### Giảm BATCH_SIZE nếu gặp lỗi Out of Memory:
```python
BATCH_SIZE = 16  # hoặc 8
```

## 🚀 Chạy ứng dụng

Sau khi training xong, chọn ứng dụng phù hợp:

**Nếu dùng PyTorch:**
```bash
python app_pytorch.py
```

**Nếu dùng TensorFlow/Keras:**
```bash
python app.py
```

Truy cập: `http://localhost:5000`

## ❓ Troubleshooting

### Lỗi: "ImageDataGenerator không khả dụng"
**Giải pháp**: Sử dụng `train_model_keras_standalone.py`

### Lỗi: "ModuleNotFoundError: No module named 'keras'"
**Giải pháp**: 
```bash
pip install keras
```

### Lỗi: "ModuleNotFoundError: No module named 'tensorflow'"
**Giải pháp**: 
```bash
pip install tensorflow
```

### Lỗi: Out of Memory
**Giải pháp**: 
1. Giảm `BATCH_SIZE` trong script
2. Giảm `IMG_SIZE` nếu cần
3. Đóng các ứng dụng khác

## 📝 Ghi chú

- **train_model_pytorch.py**: Sử dụng PyTorch, tạo model `.pth`, nhanh và hiệu quả
- **train_model.py**: Sử dụng ImageDataGenerator, phù hợp với TensorFlow/Keras đầy đủ
- **train_model_keras_standalone.py**: Xử lý dữ liệu thủ công, phù hợp khi không có ImageDataGenerator
- **Model formats**:
  - PyTorch: `.pth` files (sử dụng với `app_pytorch.py`)
  - TensorFlow/Keras: `.h5` files (sử dụng với `app.py`)
- Mỗi loại model cần ứng dụng tương ứng

## 🔄 Chuyển đổi giữa các version

Nếu gặp lỗi với một version, thử version khác:
1. Dừng script hiện tại (Ctrl+C)
2. Chạy version khác:
   - Từ `train_model.py` → `train_model_keras_standalone.py`
   - Hoặc ngược lại

## ✅ Kiểm tra sau khi training

Sau khi training xong, kiểm tra:
- `models/cat_dog_model.h5` - Model tốt nhất
- `models/cat_dog_model_final.h5` - Model cuối cùng
- `models/training_history.png` - Biểu đồ training
- `training.log` - Log chi tiết

