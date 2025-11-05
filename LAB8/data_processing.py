"""
Script xử lý dữ liệu cơ bản - Kiểm tra dữ liệu thiếu
"""
import pandas as pd
import numpy as np

def load_data(file_path):
    """Load dữ liệu từ file CSV"""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Đã tải dữ liệu từ {file_path}")
        print(f"  Số dòng: {len(df)}, Số cột: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Lỗi khi tải dữ liệu: {e}")
        return None

def check_missing_data(df):
    """Kiểm tra dữ liệu thiếu"""
    print("\n" + "="*60)
    print("KIỂM TRA DỮ LIỆU THIẾU")
    print("="*60)
    
    # Đếm số giá trị thiếu
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    # Tạo bảng tổng hợp
    missing_df = pd.DataFrame({
        'Cột': missing_count.index,
        'Số giá trị thiếu': missing_count.values,
        'Phần trăm thiếu (%)': missing_percent.values
    })
    
    # Lọc các cột có dữ liệu thiếu
    missing_df = missing_df[missing_df['Số giá trị thiếu'] > 0].sort_values('Số giá trị thiếu', ascending=False)
    
    if len(missing_df) == 0:
        print("\n✓ Không có dữ liệu thiếu!")
    else:
        print(f"\n⚠ Có {len(missing_df)} cột có dữ liệu thiếu:\n")
        print(missing_df.to_string(index=False))
    
    return missing_df

def get_data_info(df):
    """Hiển thị thông tin tổng quan về dữ liệu"""
    print("\n" + "="*60)
    print("THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU")
    print("="*60)
    
    print(f"\nKích thước dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")
    print(f"\nKiểu dữ liệu của các cột:")
    print(df.dtypes)
    
    print(f"\nThống kê mô tả (số):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe())
    
    print(f"\nThống kê mô tả (categorical):")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols[:10]:  # Giới hạn 10 cột đầu
            print(f"\n{col}:")
            print(df[col].value_counts().head())

def clean_data(df):
    """Làm sạch dữ liệu cơ bản"""
    print("\n" + "="*60)
    print("LÀM SẠCH DỮ LIỆU")
    print("="*60)
    
    initial_shape = df.shape
    
    # Loại bỏ các cột không cần thiết hoặc có quá nhiều giá trị thiếu (>50%)
    cols_to_drop = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 50:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"\nLoại bỏ {len(cols_to_drop)} cột có >50% dữ liệu thiếu:")
        for col in cols_to_drop:
            print(f"  - {col}")
        df = df.drop(columns=cols_to_drop)
    
    # Loại bỏ các dòng trùng lặp
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\nLoại bỏ {duplicates} dòng trùng lặp")
        df = df.drop_duplicates()
    
    # Điền giá trị thiếu cho các cột số bằng giá trị trung bình
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"\nĐiền giá trị thiếu cho {col} bằng giá trị trung bình: {mean_val:.2f}")
    
    # Điền giá trị thiếu cho các cột categorical bằng mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"\nĐiền giá trị thiếu cho {col} bằng mode: {mode_val}")
    
    final_shape = df.shape
    print(f"\nKích thước trước khi làm sạch: {initial_shape}")
    print(f"Kích thước sau khi làm sạch: {final_shape}")
    
    return df

def main():
    """Hàm chính"""
    print("="*60)
    print("XỬ LÝ DỮ LIỆU CƠ BẢN")
    print("="*60)
    
    # Load dữ liệu
    df_final = load_data('data/Final_data.csv')
    df_meal = load_data('data/meal_metadata.csv')
    
    if df_final is not None:
        print("\n" + "-"*60)
        print("XỬ LÝ FILE Final_data.csv")
        print("-"*60)
        check_missing_data(df_final)
        get_data_info(df_final)
        df_final_cleaned = clean_data(df_final)
        df_final_cleaned.to_csv('data/Final_data_cleaned.csv', index=False)
        print("\n✓ Đã lưu file đã làm sạch: data/Final_data_cleaned.csv")
    
    if df_meal is not None:
        print("\n" + "-"*60)
        print("XỬ LÝ FILE meal_metadata.csv")
        print("-"*60)
        check_missing_data(df_meal)
        get_data_info(df_meal)
        df_meal_cleaned = clean_data(df_meal)
        df_meal_cleaned.to_csv('data/meal_metadata_cleaned.csv', index=False)
        print("\n✓ Đã lưu file đã làm sạch: data/meal_metadata_cleaned.csv")

if __name__ == "__main__":
    main()

