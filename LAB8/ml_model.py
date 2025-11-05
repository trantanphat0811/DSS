"""
Xây dựng mô hình hồi quy tuyến tính để dự đoán BMI
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

def prepare_data(df):
    """Chuẩn bị dữ liệu cho mô hình"""
    print("\n" + "="*60)
    print("CHUẨN BỊ DỮ LIỆU")
    print("="*60)
    
    # Chọn các features có thể dùng để dự đoán BMI
    # Loại bỏ BMI_calc vì nó là công thức tính từ Weight và Height
    feature_cols = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
        'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
        'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
        'Experience_Level', 'Carbs', 'Proteins', 'Fats', 'Calories'
    ]
    
    # Lọc các cột có trong dataframe
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Tạo X và y
    X = df[available_features].copy()
    y = df['BMI'].copy()
    
    # Xử lý dữ liệu thiếu
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Loại bỏ các dòng có giá trị inf hoặc NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    print(f"\nSố features: {len(X.columns)}")
    print(f"Số mẫu: {len(X)}")
    print(f"\nFeatures được sử dụng:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
    
    return X, y

def train_model(X, y):
    """Huấn luyện mô hình hồi quy tuyến tính"""
    print("\n" + "="*60)
    print("HUẤN LUYỆN MÔ HÌNH")
    print("="*60)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDữ liệu training: {len(X_train)} mẫu")
    print(f"Dữ liệu test: {len(X_test)} mẫu")
    
    # Tạo và huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Đánh giá mô hình
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "-"*60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("-"*60)
    print(f"\nTraining Set:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    
    print(f"\nTest Set:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    
    # Hiển thị hệ số
    print("\n" + "-"*60)
    print("HỆ SỐ CỦA MÔ HÌNH (Top 10)")
    print("-"*60)
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(coef_df.head(10).to_string(index=False))
    
    return model, {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'coefficients': coef_df.to_dict('records')
    }

def save_model(model, model_path='models/bmi_model.pkl'):
    """Lưu mô hình"""
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Đã lưu mô hình: {model_path}")

def load_model(model_path='models/bmi_model.pkl'):
    """Load mô hình"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_bmi(model, features):
    """Dự đoán BMI từ các features"""
    return model.predict(features)

def main():
    """Hàm chính"""
    print("="*60)
    print("XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH - DỰ ĐOÁN BMI")
    print("="*60)
    
    # Load dữ liệu
    print("\nĐang tải dữ liệu...")
    df = pd.read_csv('data/Final_data.csv')
    print(f"✓ Đã tải {len(df)} dòng dữ liệu")
    
    # Chuẩn bị dữ liệu
    X, y = prepare_data(df)
    
    # Huấn luyện mô hình
    model, metrics = train_model(X, y)
    
    # Lưu mô hình
    save_model(model)
    
    print("\n" + "="*60)
    print("✓ Hoàn thành!")
    print("="*60)
    
    return model, metrics

if __name__ == "__main__":
    main()

