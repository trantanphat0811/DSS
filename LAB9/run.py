#!/usr/bin/env python3
"""
Script tiện ích để chạy ứng dụng
"""
import sys
import os

def main():
    print("=" * 60)
    print("Cat & Dog Classifier - Utility Script")
    print("=" * 60)
    print("\nChọn chế độ:")
    print("1. Huấn luyện mô hình (train_model.py)")
    print("2. Chạy web application (app.py)")
    print("3. Kiểm tra dependencies")
    print("4. Thoát")
    
    choice = input("\nNhập lựa chọn (1-4): ").strip()
    
    if choice == '1':
        print("\nBắt đầu huấn luyện mô hình...")
        os.system('python train_model.py')
    elif choice == '2':
        print("\nKhởi động web application...")
        print("Truy cập http://localhost:5000 sau khi server khởi động")
        os.system('python app.py')
    elif choice == '3':
        print("\nKiểm tra dependencies...")
        try:
            import flask
            import tensorflow
            import numpy
            import PIL
            import matplotlib
            print("✓ Tất cả dependencies đã được cài đặt")
            print(f"  - Flask: {flask.__version__}")
            print(f"  - TensorFlow: {tensorflow.__version__}")
            print(f"  - NumPy: {numpy.__version__}")
            print(f"  - Pillow: {PIL.__version__}")
            print(f"  - Matplotlib: {matplotlib.__version__}")
        except ImportError as e:
            print(f"✗ Thiếu dependency: {e}")
            print("  Chạy: pip install -r requirements.txt")
    elif choice == '4':
        print("Thoát...")
        sys.exit(0)
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã hủy bởi người dùng")
        sys.exit(0)

