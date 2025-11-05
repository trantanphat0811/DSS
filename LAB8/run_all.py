"""
Script chạy tất cả các bài tập một lần
"""
import subprocess
import sys
import os

def run_script(script_name, description):
    """Chạy một script và hiển thị kết quả"""
    print("\n" + "="*60)
    print(f"{description}")
    print("="*60)
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Lỗi khi chạy {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Hàm chính"""
    print("="*60)
    print("CHẠY TẤT CẢ CÁC BÀI TẬP")
    print("="*60)
    
    scripts = [
        ('data_processing.py', '1. Xử lý dữ liệu cơ bản'),
        ('visualizations.py', '2. Tạo các biểu đồ khám phá dữ liệu'),
        ('ml_model.py', '3. Xây dựng mô hình hồi quy tuyến tính')
    ]
    
    results = []
    for script, description in scripts:
        success = run_script(script, description)
        results.append((description, success))
    
    print("\n" + "="*60)
    print("TỔNG KẾT")
    print("="*60)
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {desc}")
    
    print("\n" + "="*60)
    print("Để chạy ứng dụng Flask, sử dụng lệnh:")
    print("  python app.py")
    print("="*60)

if __name__ == "__main__":
    main()

