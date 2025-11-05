"""
Script tạo các biểu đồ để khám phá dữ liệu
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Thiết lập style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
sns.set_palette("husl")

def create_output_dir():
    """Tạo thư mục lưu biểu đồ"""
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    print("✓ Đã tạo thư mục static/plots")

def plot_bmi_distribution(df):
    """Vẽ phân phối BMI"""
    plt.figure(figsize=(10, 6))
    plt.hist(df['BMI'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    plt.title('Phân phối BMI', fontsize=16, fontweight='bold')
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Tần suất', fontsize=12)
    plt.axvline(df['BMI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Trung bình: {df["BMI"].mean():.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/bmi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ phân phối BMI")

def plot_bmi_by_gender(df):
    """Vẽ BMI theo giới tính"""
    plt.figure(figsize=(10, 6))
    df.boxplot(column='BMI', by='Gender', figsize=(10, 6))
    plt.title('BMI theo Giới tính', fontsize=16, fontweight='bold')
    plt.suptitle('')  # Loại bỏ title mặc định
    plt.xlabel('Giới tính', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/bmi_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ BMI theo giới tính")

def plot_bmi_by_workout_type(df):
    """Vẽ BMI theo loại tập luyện"""
    plt.figure(figsize=(12, 6))
    workout_order = df.groupby('Workout_Type')['BMI'].mean().sort_values().index
    sns.boxplot(data=df, x='Workout_Type', y='BMI', order=workout_order)
    plt.title('BMI theo Loại Tập luyện', fontsize=16, fontweight='bold')
    plt.xlabel('Loại Tập luyện', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/bmi_by_workout_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ BMI theo loại tập luyện")

def plot_correlation_heatmap(df):
    """Vẽ bản đồ tương quan"""
    # Chọn các cột số liên quan đến BMI
    numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Calories_Burned', 
                    'Fat_Percentage', 'Workout_Frequency (days/week)', 'Calories', 
                    'Carbs', 'Proteins', 'Fats']
    
    # Lọc các cột có trong dataframe
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_df = df[available_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Bản đồ Tương quan giữa các Biến', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo bản đồ tương quan")

def plot_age_vs_bmi(df):
    """Vẽ mối quan hệ giữa Tuổi và BMI"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Age'], df['BMI'], alpha=0.5, s=20)
    
    # Vẽ đường hồi quy
    z = np.polyfit(df['Age'].dropna(), df['BMI'].dropna(), 1)
    p = np.poly1d(z)
    plt.plot(df['Age'], p(df['Age']), "r--", linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    plt.title('Mối quan hệ giữa Tuổi và BMI', fontsize=16, fontweight='bold')
    plt.xlabel('Tuổi', fontsize=12)
    plt.ylabel('BMI', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/plots/age_vs_bmi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ Tuổi vs BMI")

def plot_weight_height_bmi(df):
    """Vẽ mối quan hệ Weight, Height và BMI"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Weight vs BMI
    axes[0].scatter(df['Weight (kg)'], df['BMI'], alpha=0.5, s=20)
    z1 = np.polyfit(df['Weight (kg)'].dropna(), df['BMI'].dropna(), 1)
    p1 = np.poly1d(z1)
    axes[0].plot(df['Weight (kg)'], p1(df['Weight (kg)']), "r--", linewidth=2)
    axes[0].set_title('Weight vs BMI', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Weight (kg)', fontsize=12)
    axes[0].set_ylabel('BMI', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Height vs BMI
    axes[1].scatter(df['Height (m)'], df['BMI'], alpha=0.5, s=20)
    z2 = np.polyfit(df['Height (m)'].dropna(), df['BMI'].dropna(), 1)
    p2 = np.poly1d(z2)
    axes[1].plot(df['Height (m)'], p2(df['Height (m)']), "r--", linewidth=2)
    axes[1].set_title('Height vs BMI', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Height (m)', fontsize=12)
    axes[1].set_ylabel('BMI', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/plots/weight_height_bmi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ Weight/Height vs BMI")

def plot_calories_vs_bmi(df):
    """Vẽ mối quan hệ Calories và BMI"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calories consumed vs BMI
    axes[0].scatter(df['Calories'], df['BMI'], alpha=0.5, s=20)
    axes[0].set_title('Calories Tiêu thụ vs BMI', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Calories Tiêu thụ', fontsize=12)
    axes[0].set_ylabel('BMI', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Calories burned vs BMI
    axes[1].scatter(df['Calories_Burned'], df['BMI'], alpha=0.5, s=20)
    axes[1].set_title('Calories Đốt cháy vs BMI', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Calories Đốt cháy', fontsize=12)
    axes[1].set_ylabel('BMI', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/plots/calories_vs_bmi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Đã tạo biểu đồ Calories vs BMI")

def plot_diet_type_bmi(df):
    """Vẽ BMI theo loại chế độ ăn"""
    if 'diet_type' in df.columns:
        plt.figure(figsize=(12, 6))
        diet_order = df.groupby('diet_type')['BMI'].mean().sort_values().index
        sns.boxplot(data=df, x='diet_type', y='BMI', order=diet_order)
        plt.title('BMI theo Loại Chế độ Ăn', fontsize=16, fontweight='bold')
        plt.xlabel('Loại Chế độ Ăn', fontsize=12)
        plt.ylabel('BMI', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/plots/bmi_by_diet_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Đã tạo biểu đồ BMI theo loại chế độ ăn")

def main():
    """Hàm chính"""
    print("="*60)
    print("TẠO CÁC BIỂU ĐỒ KHÁM PHÁ DỮ LIỆU")
    print("="*60)
    
    # Tạo thư mục output
    create_output_dir()
    
    # Load dữ liệu
    print("\nĐang tải dữ liệu...")
    df = pd.read_csv('data/Final_data.csv')
    print(f"✓ Đã tải {len(df)} dòng dữ liệu")
    
    # Tạo các biểu đồ
    print("\nĐang tạo các biểu đồ...")
    plot_bmi_distribution(df)
    plot_bmi_by_gender(df)
    plot_bmi_by_workout_type(df)
    plot_correlation_heatmap(df)
    plot_age_vs_bmi(df)
    plot_weight_height_bmi(df)
    plot_calories_vs_bmi(df)
    plot_diet_type_bmi(df)
    
    print("\n" + "="*60)
    print("✓ Hoàn thành! Tất cả biểu đồ đã được lưu trong thư mục static/plots")
    print("="*60)

if __name__ == "__main__":
    main()

