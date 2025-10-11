import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = {
    "MSSV": ["SV001", "SV002", "SV003", "SV004", "SV005", "SV006", "SV007", "SV008", "SV009", "SV010"],
    "Ho Va Ten": ["Nguyen Van A", "Tran Thi B", "Le Van C", "Pham Thi D", "Hoang Van E", 
                  "Vu Thi F", "Do Van G", "Bui Thi H", "Dang Van I", "Ly Thi J"],
    "Gioi Tinh": ["Nam", "Nu", "Nam", "Nu", "Nam", "Nu", "Nam", "Nu", "Nam", "Nu"],
    "Diem Toan": [8.5, 7.0, np.nan, 9.2, 6.5, 8.0, 7.8, 9.5, 5.5, 8.8],
    "Diem Van": [7.5, 8.5, 6.0, np.nan, 7.0, 9.0, 8.2, 7.8, 6.5, 9.2],
    "Diem Anh": [9.0, 6.5, 8.0, 7.5, np.nan, 8.5, 9.2, 6.0, 7.0, 8.0],
    "Tuoi": [20, 21, 19, 22, 20, 21, 20, 19, 22, 21]
}
df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)

df["Diem Toan"] = df["Diem Toan"].fillna(df["Diem Toan"].mean())
df["Diem Van"] = df["Diem Van"].fillna(df["Diem Van"].mean())
df["Diem Anh"] = df["Diem Anh"].fillna(df["Diem Anh"].mean())
print(df)
avg_scores = df.groupby("Gioi Tinh")[["Diem Toan", "Diem Van", "Diem Anh"]].mean()
avg_scores.plot(kind='bar')
plt.title("Điểm trung bình theo giới tính")
plt.ylabel("Điểm trung bình")
plt.xlabel("Giới tính")
plt.legend(title="Môn học")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.scatter(df["Diem Toan"], df["Diem Anh"], color='blue', alpha=0.7)
plt.title("Biểu đồ phân tán: Điểm Toán vs Điểm Anh")
plt.xlabel("Điểm Toán")
plt.ylabel("Điểm Anh")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

df = df.dropna(subset=["Diem Toan","Diem Van","Diem Anh"])

X = df[["Diem Toan","Diem Van"]]
y = df["Diem Anh"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("=== ĐÁNH GIÁ MÔ HÌNH HỒI QUY TUYẾN TÍNH ===")
print("Hệ số hồi quy (weights):", model.coef_)
print("Hệ số chặn (intercept):", model.intercept_)
print("Mean Squared Error (MSE):", round(mse, 4))
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Giá trị thực tế (y_test)")
plt.ylabel("Giá trị dự đoán (y_pred)")
plt.title("So sánh dự đoán và giá trị thực tế")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

