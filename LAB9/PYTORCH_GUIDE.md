# Hướng dẫn sử dụng PyTorch Version

## 🎯 Tại sao sử dụng PyTorch?

PyTorch là một thư viện deep learning mạnh mẽ, được nhiều nhà nghiên cứu và developer sử dụng vì:
- ✅ Tốc độ nhanh, đặc biệt với GPU
- ✅ Code dễ đọc và debug
- ✅ Hỗ trợ dynamic computation graphs
- ✅ Cộng đồng lớn và tài liệu tốt
- ✅ Hoàn toàn thay thế được TensorFlow/Keras

## 📦 Cài đặt

### Cài đặt PyTorch:

```bash
# CPU version (mặc định)
pip install torch torchvision

# GPU version (nếu có NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Kiểm tra cài đặt:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🚀 Sử dụng

### 1. Huấn luyện mô hình:

```bash
python train_model_pytorch.py
```

**Lưu ý**:
- Quá trình training sẽ tự động sử dụng GPU nếu có
- Model sẽ được lưu vào `models/cat_dog_model_pytorch.pth`
- Biểu đồ training: `models/training_history_pytorch.png`

### 2. Chạy web application:

```bash
python app_pytorch.py
```

Truy cập: `http://localhost:5000`

## 📊 So sánh với TensorFlow/Keras

| Tính năng | PyTorch | TensorFlow/Keras |
|-----------|---------|------------------|
| Tốc độ training | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Dễ sử dụng | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| GPU support | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Community | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model format | .pth | .h5 |

## 🔧 Cấu hình

### Thay đổi tham số trong `train_model_pytorch.py`:

```python
IMG_SIZE = 150        # Kích thước ảnh
BATCH_SIZE = 32       # Batch size
EPOCHS = 10           # Số epochs
LEARNING_RATE = 1e-4  # Learning rate
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Sử dụng GPU:

Mô hình sẽ tự động sử dụng GPU nếu có. Để kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # True nếu có GPU
```

## 📁 Files được tạo

Sau khi training, các file sau sẽ được tạo:
- `models/cat_dog_model_pytorch.pth` - Best model (state_dict)
- `models/cat_dog_model_pytorch_final.pth` - Final model (state_dict)
- `models/cat_dog_model_pytorch_full.pth` - Full model (dễ load)
- `models/training_history_pytorch.png` - Biểu đồ training

## 🔄 Migration từ TensorFlow/Keras

Nếu đang dùng TensorFlow/Keras và muốn chuyển sang PyTorch:

1. **Training**:
   ```bash
   # Cũ
   python train_model.py
   
   # Mới
   python train_model_pytorch.py
   ```

2. **Application**:
   ```bash
   # Cũ
   python app.py
   
   # Mới
   python app_pytorch.py
   ```

3. **Model files**: Không tương thích, cần train lại

## ❓ Troubleshooting

### Lỗi: "CUDA out of memory"
**Giải pháp**: Giảm `BATCH_SIZE` trong script

### Lỗi: "ModuleNotFoundError: No module named 'torch'"
**Giải pháp**: 
```bash
pip install torch torchvision
```

### Lỗi: Model không load được
**Giải pháp**: Đảm bảo model class giống với training script

## 💡 Tips

1. **GPU**: Sử dụng GPU sẽ nhanh hơn 10-100 lần
2. **Batch size**: Tăng batch size nếu có nhiều RAM/GPU memory
3. **Mixed precision**: Có thể sử dụng để tăng tốc độ
4. **DataLoader**: Tăng `num_workers` nếu có nhiều CPU cores

## 📚 Tài liệu tham khảo

- PyTorch official: https://pytorch.org/
- PyTorch tutorials: https://pytorch.org/tutorials/
- torchvision: https://pytorch.org/vision/stable/index.html

---

**Lưu ý**: Phiên bản PyTorch hoàn toàn độc lập với TensorFlow/Keras version. Bạn có thể sử dụng một trong hai, hoặc cả hai tùy nhu cầu.

