# 🛡️ Spam Email Detection System

Ứng dụng web phân loại thư rác sử dụng thuật toán Naive Bayes được xây dựng với Flask.

## 📋 Mô tả

Hệ thống này sử dụng thuật toán Naive Bayes để phân loại email thành spam hoặc không spam (ham). Ứng dụng có giao diện web thân thiện cho phép người dùng nhập nội dung email và nhận kết quả phân loại ngay lập tức.

## 🚀 Tính năng

- ✅ Phân loại email spam/ham sử dụng Naive Bayes
- ✅ Giao diện web responsive và thân thiện
- ✅ Hiển thị độ tin cậy của kết quả
- ✅ Ví dụ mẫu để test
- ✅ API endpoint để tích hợp với hệ thống khác
- ✅ Error handling và logging tốt
- ✅ Cấu trúc code modular (CSS/JS riêng biệt)
- ✅ Validation đầu vào và giới hạn độ dài
- ✅ Dashboard analytics với Plotly charts
- ✅ Biểu đồ trực quan hóa dữ liệu
- ✅ Thống kê và phân tích hiệu suất mô hình
- ✅ Error handling JavaScript robust
- ✅ DOM elements initialization an toàn

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd Lab7
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Chạy ứng dụng

```bash
python app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:5001`

## 📁 Cấu trúc thư mục

```
Lab7/
├── app.py                 # Ứng dụng Flask chính
├── requirements.txt       # Dependencies
├── README.md             # Hướng dẫn này
├── data/
│   └── emails.csv        # Dữ liệu training
├── static/
│   ├── css/
│   │   └── style.css     # CSS styles
│   └── js/
│       └── main.js       # JavaScript logic
└── templates/
    └── index.html        # Giao diện web
```

## 🔧 Cách sử dụng

1. Mở trình duyệt và truy cập `http://localhost:5001`
2. Nhập nội dung email vào ô textarea
3. Nhấn nút "Kiểm Tra Email"
4. Xem kết quả phân loại và độ tin cậy
5. Truy cập `/dashboard` để xem analytics và biểu đồ

### Ví dụ email spam:
```
WIN FREE MONEY NOW! Click here to claim your $1000 prize! Limited time offer!
```

### Ví dụ email bình thường:
```
Hi John, I hope you're doing well. I wanted to follow up on our meeting yesterday.
```

## 🛠️ API Endpoints

### POST /predict
Dự đoán email có phải spam hay không.

**Request:**
```json
{
    "email_text": "Nội dung email cần kiểm tra"
}
```

**Response:**
```json
{
    "success": true,
    "result": "SPAM",
    "confidence": 85.67,
    "email_text": "Nội dung email cần kiểm tra"
}
```

### GET /health
Kiểm tra trạng thái ứng dụng.

**Response:**
```json
{
    "status": "OK",
    "model_loaded": true,
    "vectorizer_loaded": true
}
```

### GET /dashboard
Trang dashboard với biểu đồ analytics.

### GET /api/charts/data-distribution
Lấy dữ liệu biểu đồ phân phối spam/ham.

**Response:** JSON data cho Plotly pie chart

### GET /api/charts/top-words
Lấy dữ liệu biểu đồ từ khóa spam phổ biến.

**Response:** JSON data cho Plotly bar chart

### GET /api/charts/model-performance
Lấy dữ liệu biểu đồ hiệu suất mô hình (confusion matrix).

**Response:** JSON data cho Plotly heatmap

## 📊 Dữ liệu Training

Dữ liệu training được lưu trong file `data/emails.csv` với cấu trúc:
- Cột đầu tiên: Email No. (ID email)
- Các cột giữa: Tần suất xuất hiện của các từ
- Cột cuối: Prediction (0 = ham, 1 = spam)

## 🧠 Thuật toán

Ứng dụng sử dụng **Multinomial Naive Bayes** từ scikit-learn:
- Phù hợp với dữ liệu text đã được vector hóa
- Hiệu quả cao với dữ liệu thưa (sparse data)
- Tốc độ xử lý nhanh

## 🔍 Xử lý văn bản

1. **Tiền xử lý:** Chuyển về chữ thường, loại bỏ ký tự đặc biệt
2. **Vectorization:** Sử dụng CountVectorizer với vocabulary có sẵn
3. **Dự đoán:** Áp dụng mô hình Naive Bayes đã được huấn luyện

## 🐛 Troubleshooting

### Lỗi "Mô hình chưa được tải"
- Kiểm tra file `data/emails.csv` có tồn tại
- Đảm bảo dữ liệu có đúng format
- Kiểm tra logs trong terminal

### Lỗi kết nối
- Đảm bảo ứng dụng đang chạy trên port 5001
- Kiểm tra firewall settings
- Thử truy cập `http://127.0.0.1:5001`

## 📈 Cải tiến có thể

- [ ] Thêm tính năng upload file email
- [ ] Hỗ trợ nhiều ngôn ngữ
- [ ] Tích hợp với email client
- [ ] Dashboard thống kê
- [ ] Model retraining tự động

## 📝 License

MIT License

## 👨‍💻 Tác giả

Tạo bởi AI Assistant cho môn DSS Lab 7
