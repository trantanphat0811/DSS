"""
Ứng dụng Flask để trực quan hóa Dashboard và mô hình học máy
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from ml_model import prepare_data, load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load dữ liệu và mô hình (nhẹ, tránh log/thao tác dư thừa)
data_path = 'data/Final_data_cleaned.csv' if os.path.exists('data/Final_data_cleaned.csv') else 'data/Final_data.csv'
df = pd.read_csv(data_path)

# Chuẩn bị dữ liệu cho mô hình
X, y = prepare_data(df)
feature_columns = list(X.columns)

# Load mô hình; không tự huấn luyện để tránh chặn khởi động server
model_path = 'models/bmi_model.pkl'
model = load_model(model_path) if os.path.exists(model_path) else None

# Tự động huấn luyện nếu chưa có mô hình
if model is None:
    try:
        from ml_model import train_model, save_model
        model, _metrics = train_model(X, y)
        save_model(model, model_path)
    except Exception as e:
        # Cho phép server vẫn chạy, nhưng các API phụ thuộc model sẽ báo 503
        model = None

# Cache sẵn thống kê và thông tin mô hình để API phản hồi nhanh
_stats_cache = {
    'total_records': int(len(df)),
    'avg_bmi': float(round(df['BMI'].mean(), 2)),
    'avg_age': float(round(df['Age'].mean(), 2)),
    'avg_weight': float(round(df['Weight (kg)'].mean(), 2)),
    'avg_height': float(round(df['Height (m)'].mean(), 2)),
    'gender_dist': df['Gender'].value_counts().to_dict(),
    'workout_types': df['Workout_Type'].value_counts().to_dict(),
    'diet_types': df['diet_type'].value_counts().to_dict() if 'diet_type' in df.columns else {}
}

_bmi_by_category_cache = {
    'by_gender': df.groupby('Gender')['BMI'].mean().to_dict(),
    'by_workout': df.groupby('Workout_Type')['BMI'].mean().to_dict(),
    'by_diet': df.groupby('diet_type')['BMI'].mean().to_dict() if 'diet_type' in df.columns else {}
}

# Correlation cache for Plotly heatmap
_corr_cols = [
    'Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Calories_Burned',
    'Fat_Percentage', 'Workout_Frequency (days/week)', 'Calories',
    'Carbs', 'Proteins', 'Fats'
]
_corr_cols = [c for c in _corr_cols if c in df.columns]
_corr_matrix = df[_corr_cols].corr().round(3)

_model_info_cache = None
if model is not None:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_pred_full = model.predict(X)
    r2 = float(round(r2_score(y, y_pred_full), 4))
    rmse = float(round(np.sqrt(mean_squared_error(y, y_pred_full)), 4))
    mae = float(round(mean_absolute_error(y, y_pred_full), 4))
    coef_df = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    _model_info_cache = {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'top_features': coef_df.head(10).to_dict('records'),
        'intercept': float(round(model.intercept_, 4))
    }

@app.route('/')
def index():
    """Trang chủ - Dashboard"""
    return render_template('index.html')

@app.route('/predict-page')
def predict_page():
    """Trang dự đoán BMI (tách riêng)"""
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API dự đoán BMI"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model chưa sẵn sàng. Vui lòng chạy: python ml_model.py'}), 503
        data = request.json
        
        # Tạo dataframe từ input
        input_data = {}
        for col in feature_columns:
            if col in data:
                input_data[col] = [data[col]]
            else:
                # Giá trị mặc định nếu thiếu
                input_data[col] = [df[col].mean() if col in df.columns else 0]
        
        input_df = pd.DataFrame(input_data)
        
        # Dự đoán
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'predicted_bmi': round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/stats')
def get_stats():
    """API lấy thống kê tổng quan"""
    return jsonify(_stats_cache)

@app.route('/api/bmi_by_category')
def get_bmi_by_category():
    """API lấy BMI theo các danh mục"""
    return jsonify(_bmi_by_category_cache)

@app.route('/api/correlation')
def get_correlation():
    """API trả về ma trận tương quan cho heatmap Plotly"""
    return jsonify({
        'labels': _corr_cols,
        'matrix': _corr_matrix.values.tolist()
    })

@app.route('/api/model_info')
def get_model_info():
    """API lấy thông tin về mô hình"""
    if _model_info_cache is None:
        return jsonify({'error': 'Model chưa sẵn sàng. Vui lòng chạy: python ml_model.py'}), 503
    return jsonify(_model_info_cache)

if __name__ == '__main__':
    # Tạo các thư mục cần thiết
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Cấu hình cổng linh hoạt, tránh xung đột cổng 5000 mặc định
    port = int(os.environ.get('PORT', '5050'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"Server running on http://127.0.0.1:{port}")
    app.run(debug=debug, host='0.0.0.0', port=port)

