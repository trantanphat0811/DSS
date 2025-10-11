#   LAB 5: Xử lý dữ liệu sinh viên và Hồi quy tuyến tính

##  BÀI 1: Xử lý dữ liệu thiếu bằng mean()
```python
df["Diem Toan"] = df["Diem Toan"].fillna(df["Diem Toan"].mean())
df["Diem Van"] = df["Diem Van"].fillna(df["Diem Van"].mean())
df["Diem Anh"] = df["Diem Anh"].fillna(df["Diem Anh"].mean())
```
👉 Thay giá trị thiếu bằng giá trị trung bình cột tương ứng. Tránh dùng `inplace=True` để không báo lỗi pandas 3.0.

---

##  BÀI 2: Biểu đồ cột – Điểm trung bình theo Giới tính
```python
import matplotlib.pyplot as plt
df.groupby("Gioi Tinh")[["Diem Toan", "Diem Van", "Diem Anh"]].mean().plot(kind='bar')
plt.title("Điểm trung bình theo Giới tính")
plt.xlabel("Giới tính")
plt.ylabel("Điểm trung bình")
plt.show()
```

---

##  BÀI 3: Biểu đồ phân tán (Scatter Plot) – Điểm Toán vs Điểm Anh
```python
plt.scatter(df["Diem Toan"], df["Diem Anh"], color='blue')
plt.title("Biểu đồ phân tán: Điểm Toán vs Điểm Anh")
plt.xlabel("Điểm Toán")
plt.ylabel("Điểm Anh")
plt.show()
```

---

##  BÀI 4: Hồi quy tuyến tính (Linear Regression)
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_clean = df.dropna(subset=["Diem Toan", "Diem Van", "Diem Anh"])
X = df_clean[["Diem Toan", "Diem Van"]]
y = df_clean["Diem Anh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Hệ số hồi quy:", model.coef_)
print("Sai số trung bình bình phương (MSE):", mse)


---

##  HƯỚNG DẪN SỬ DỤNG NANO TRÊN MÁY ẢO LINUX

### 1️ Mở terminal
```bash
Ctrl + Alt + T
```

###  Tạo thư mục
```bash
mkdir -p ~/iDragonCloud
cd ~/iDragonCloud
```

###  Tạo file Python mới
```bash
nano lab5.py


###  Chạy file Python

python3 lab5.py



