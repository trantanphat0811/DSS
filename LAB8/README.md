# LAB8 - Phân tích dữ liệu BMI và Dự đoán

Dự án này bao gồm 5 bài tập chính:
1. Xử lý dữ liệu cơ bản bằng Python (kiểm tra dữ liệu thiếu)
2. Vẽ các biểu đồ để khám phá dữ liệu
3. Xây dựng dashboard đơn giản
4. Sử dụng mô hình hồi quy tuyến tính để dự đoán chỉ số BMI
5. Sử dụng Flask để trực quan hóa Dashboard và mô hình học máy lên web

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
LAB8/
├── data/
│   ├── Final_data.csv          # Dữ liệu chính
│   └── meal_metadata.csv        # Metadata về bữa ăn
├── data_processing.py           # Script xử lý dữ liệu
├── visualizations.py            # Script tạo biểu đồ
├── ml_model.py                  # Mô hình học máy
├── app.py                       # Ứng dụng Flask
├── templates/
│   └── index.html              # Template HTML cho dashboard
├── static/
│   └── plots/                  # Thư mục lưu biểu đồ (tự động tạo)
├── models/                     # Thư mục lưu mô hình (tự động tạo)
└── requirements.txt            # Các thư viện cần thiết
```

## Hướng dẫn sử dụng

### 1. Xử lý dữ liệu cơ bản

Chạy script để kiểm tra và xử lý dữ liệu thiếu:
```bash
python data_processing.py
```

Script này sẽ:
- Kiểm tra dữ liệu thiếu trong cả 2 file
- Hiển thị thống kê tổng quan về dữ liệu
- Làm sạch dữ liệu (điền giá trị thiếu, loại bỏ trùng lặp)
- Lưu file đã làm sạch

### 2. Tạo biểu đồ khám phá dữ liệu

Chạy script để tạo các biểu đồ:
```bash
python visualizations.py
```

Các biểu đồ sẽ được lưu trong thư mục `static/plots/`:
- Phân phối BMI
- BMI theo giới tính
- BMI theo loại tập luyện
- Bản đồ tương quan
- Tuổi vs BMI
- Weight/Height vs BMI
- Calories vs BMI
- BMI theo loại chế độ ăn

### 3. Xây dựng mô hình học máy

Chạy script để huấn luyện mô hình:
```bash
python ml_model.py
```

Mô hình sẽ được lưu trong `models/bmi_model.pkl`. Script sẽ hiển thị:
- Các metrics đánh giá mô hình (R², RMSE, MAE)
- Top 10 features quan trọng nhất

### 4. Chạy ứng dụng Flask

Khởi động server Flask:
```bash
python app.py
```

Sau đó mở trình duyệt và truy cập:
```
http://127.0.0.1:5000
```

## Tính năng Dashboard

Dashboard bao gồm:

1. **Thống kê tổng quan**: Hiển thị số lượng bản ghi, BMI trung bình, tuổi trung bình, cân nặng trung bình

2. **Dự đoán BMI**: Form nhập thông tin để dự đoán BMI:
   - Tuổi, Cân nặng, Chiều cao (bắt buộc)
   - Max BPM, Avg BPM, Resting BPM
   - Calories Burned, Fat Percentage
   - Calories, Carbs, Proteins, Fats

3. **Thông tin mô hình**: Hiển thị các metrics và top features quan trọng

4. **Biểu đồ tương tác**:
   - BMI theo giới tính
   - BMI theo loại tập luyện
   - Bản đồ tương quan

## API Endpoints

- `GET /`: Trang chủ dashboard
- `POST /predict`: Dự đoán BMI từ dữ liệu đầu vào
- `GET /api/stats`: Lấy thống kê tổng quan
- `GET /api/bmi_by_category`: Lấy BMI theo các danh mục
- `GET /api/model_info`: Lấy thông tin về mô hình

## Ví dụ sử dụng API

### Dự đoán BMI:
```python
import requests
import json

data = {
    "Age": 30,
    "Weight (kg)": 70,
    "Height (m)": 1.75,
    "Calories_Burned": 500,
    "Calories": 2000
}

response = requests.post('http://127.0.0.1:5000/predict', json=data)
result = response.json()
print(f"Predicted BMI: {result['predicted_bmi']}")
```

## Lưu ý

- Đảm bảo các file dữ liệu nằm trong thư mục `data/`
- Thư mục `static/plots/` và `models/` sẽ được tạo tự động khi chạy các script
- Nếu mô hình chưa được train, ứng dụng Flask sẽ tự động train khi khởi động

## Tác giả

Dự án được phát triển cho LAB8 - Phân tích dữ liệu và Machine Learning

