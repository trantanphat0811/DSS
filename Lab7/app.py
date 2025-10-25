from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objs as go
import plotly.utils
import json
import re
import os
import logging

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo các biến global
model = None
vectorizer = None
feature_names = None
df = None
X_train = None
X_test = None
y_train = None
y_test = None

def preprocess_text(text):
    """Tiền xử lý văn bản"""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt và số
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data():
    """Tải và chuẩn bị dữ liệu"""
    global model, vectorizer, feature_names, df, X_train, X_test, y_train, y_test
    
    try:
        # Đọc dữ liệu
        df = pd.read_csv('data/emails.csv')
        
        # Lấy các cột feature (bỏ cột đầu tiên Email No. và cột cuối Prediction)
        feature_columns = df.columns[1:-1]  # Bỏ Email No. và Prediction
        X = df[feature_columns].values
        y = df['Prediction'].values
        
        # Lưu tên các feature để sử dụng sau
        feature_names = feature_columns.tolist()
        
        # Chia dữ liệu train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Huấn luyện mô hình Naive Bayes
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Tạo vectorizer để xử lý văn bản mới
        vectorizer = CountVectorizer(vocabulary=feature_names, lowercase=True)
        
        logger.info(f"Mô hình đã được huấn luyện với {len(X_train)} mẫu")
        logger.info(f"Độ chính xác trên tập test: {model.score(X_test, y_test):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        return False

def predict_spam(text):
    """Dự đoán email có phải spam hay không"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        return None, "Mô hình chưa được khởi tạo"
    
    try:
        # Tiền xử lý văn bản
        processed_text = preprocess_text(text)
        
        # Vectorize văn bản
        text_vector = vectorizer.transform([processed_text])
        
        # Dự đoán
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Trả về kết quả
        result = "SPAM" if prediction == 1 else "KHÔNG PHẢI SPAM"
        confidence = max(probability) * 100
        
        return result, confidence
        
    except Exception as e:
        return None, f"Lỗi khi dự đoán: {e}"

def create_data_distribution_chart():
    """Tạo biểu đồ phân phối dữ liệu spam/ham"""
    global df
    
    if df is None:
        return None
    
    # Đếm số lượng spam và ham
    spam_count = len(df[df['Prediction'] == 1])
    ham_count = len(df[df['Prediction'] == 0])
    
    # Tạo biểu đồ pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['Ham (Không spam)', 'Spam'],
        values=[ham_count, spam_count],
        hole=0.3,
        marker_colors=['#4CAF50', '#f44336']
    )])
    
    fig.update_layout(
        title="Phân phối Email trong Dataset",
        font=dict(size=14),
        showlegend=True,
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_top_words_chart():
    """Tạo biểu đồ từ khóa xuất hiện nhiều nhất"""
    global df, feature_names
    
    if df is None or feature_names is None:
        return None
    
    # Tính tổng tần suất của mỗi từ trong spam emails
    spam_emails = df[df['Prediction'] == 1]
    spam_word_counts = spam_emails[feature_names].sum().sort_values(ascending=False).head(20)
    
    # Tạo biểu đồ bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=spam_word_counts.values,
            y=spam_word_counts.index,
            orientation='h',
            marker_color='#f44336'
        )
    ])
    
    fig.update_layout(
        title="Top 20 Từ Khóa Xuất Hiện Nhiều Nhất trong Email Spam",
        xaxis_title="Tần suất xuất hiện",
        yaxis_title="Từ khóa",
        font=dict(size=12),
        height=600
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_model_performance_chart():
    """Tạo biểu đồ hiệu suất mô hình"""
    global model, X_test, y_test
    
    if model is None or X_test is None or y_test is None:
        return None
    
    # Dự đoán trên tập test
    y_pred = model.predict(X_test)
    
    # Tính các metric
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    
    # Tạo biểu đồ confusion matrix
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Dự đoán Ham', 'Dự đoán Spam'],
        y=['Thực tế Ham', 'Thực tế Spam'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        font=dict(size=14),
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để dự đoán spam"""
    try:
        # Kiểm tra Content-Type
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type phải là application/json'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Dữ liệu JSON không hợp lệ'
            }), 400
            
        email_text = data.get('email_text', '')
        
        if not email_text.strip():
            return jsonify({
                'success': False,
                'error': 'Vui lòng nhập nội dung email'
            }), 400
        
        # Giới hạn độ dài email
        if len(email_text) > 10000:
            return jsonify({
                'success': False,
                'error': 'Email quá dài (tối đa 10,000 ký tự)'
            }), 400
        
        logger.info(f"Đang xử lý email có độ dài: {len(email_text)} ký tự")
        result, confidence = predict_spam(email_text)
        
        if result is None:
            logger.error(f"Lỗi khi dự đoán: {confidence}")
            return jsonify({
                'success': False,
                'error': confidence
            }), 500
        
        logger.info(f"Kết quả dự đoán: {result} với độ tin cậy {confidence:.2f}%")
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 2),
            'email_text': email_text[:100] + '...' if len(email_text) > 100 else email_text
        })
        
    except Exception as e:
        logger.error(f"Lỗi server trong predict: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Kiểm tra trạng thái ứng dụng"""
    return jsonify({
        'status': 'OK',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

@app.route('/dashboard')
def dashboard():
    """Trang dashboard với biểu đồ"""
    return render_template('dashboard.html')

@app.route('/api/charts/data-distribution')
def get_data_distribution():
    """API endpoint để lấy dữ liệu biểu đồ phân phối"""
    try:
        chart_data = create_data_distribution_chart()
        if chart_data:
            return chart_data
        else:
            return jsonify({'error': 'Không thể tạo biểu đồ'}), 500
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ phân phối: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/top-words')
def get_top_words():
    """API endpoint để lấy biểu đồ từ khóa"""
    try:
        chart_data = create_top_words_chart()
        if chart_data:
            return chart_data
        else:
            return jsonify({'error': 'Không thể tạo biểu đồ'}), 500
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ từ khóa: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/model-performance')
def get_model_performance():
    """API endpoint để lấy biểu đồ hiệu suất mô hình"""
    try:
        chart_data = create_model_performance_chart()
        if chart_data:
            return chart_data
        else:
            return jsonify({'error': 'Không thể tạo biểu đồ'}), 500
    except Exception as e:
        logger.error(f"Lỗi khi tạo biểu đồ hiệu suất: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tạo thư mục cần thiết nếu chưa có
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    # Khởi tạo mô hình
    logger.info("Đang khởi tạo mô hình...")
    if load_and_prepare_data():
        logger.info("Mô hình đã sẵn sàng!")
        logger.info("Ứng dụng đang chạy tại: http://localhost:5001")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("Không thể khởi tạo mô hình. Vui lòng kiểm tra dữ liệu.")
