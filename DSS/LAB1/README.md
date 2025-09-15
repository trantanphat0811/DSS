# 📌 README – Thực hành với Dataiku và dataset `orders.csv`

## 1. Giới thiệu
Trong bài tập này, chúng ta sẽ thực hành với **Dataiku DSS** để:
- Import dữ liệu từ file CSV (`orders.csv`) vào Dataiku.  
- Thực hiện làm sạch dữ liệu (rename cột, chuyển đổi kiểu dữ liệu).  
- Khám phá dữ liệu bằng các chức năng **Explore, Charts, Statistics**.  
- Chuẩn bị dữ liệu cho phân tích và xây dựng mô hình.  

---

## 2. Các bước thực hiện

### Bước 1: Tạo Project
1. Đăng nhập vào **Dataiku DSS**.  
2. Chọn **+ New Project → Blank Project**.  
3. Đặt tên project, ví dụ: `LAB1_orders`.  

---

### Bước 2: Import dữ liệu CSV
1. Vào giao diện **Flow**.  
2. Chọn **Connect or create → Upload your files**.  
3. Upload file `orders.csv`.  
4. Sau khi upload, dataset sẽ xuất hiện trong **Flow** với tên `orders`.  

---

### Bước 3: Khám phá dữ liệu
1. Mở dataset `orders`.  
2. Kiểm tra dữ liệu trong tab **Explore**:
   - Xem số lượng dòng.  
   - Kiểm tra các giá trị bị thiếu (null).  
   - Kiểm tra kiểu dữ liệu từng cột.  

---

### Bước 4: Chuẩn hóa dữ liệu
1. **Đổi tên cột** về dạng chuẩn (snake_case).  
   Ví dụ:  
   - `Order ID` → `order_id`  
   - `Customer Name` → `customer_name`  
   - `Order Date` → `order_date`  
   - `Sales` → `sales_vnd`  

2. **Chuyển kiểu dữ liệu**:  
   - `order_date` → Date  
   - `sales_vnd` → Float/Integer  
   - Các cột text (`customer_name`, `product_name`, …) → String  

3. Thực hiện bằng **Prepare recipe** trong Flow.  

---

### Bước 5: Trực quan hóa dữ liệu
1. Chọn tab **Charts** trong dataset.  
2. Vẽ biểu đồ ví dụ:  
   - Tổng doanh thu (`sales_vnd`) theo năm (`order_date`).  
   - Doanh thu theo từng khách hàng.  
   - Top 10 sản phẩm bán chạy nhất.  

---

### Bước 6: Phân tích thống kê
1. Chọn tab **Statistics**.  
2. Xem:  
   - Phân bố dữ liệu (distribution) của `sales_vnd`.  
   - Thống kê mô tả (min, max, mean, median).  
   - Số lượng đơn hàng theo nhóm khách hàng.  

---

### Bước 7: Chuẩn bị cho mô hình (tuỳ chọn nâng cao)
1. Dùng **Prepare recipe** để:  
   - Loại bỏ dòng null.  
   - Tạo cột mới, ví dụ `year = YEAR(order_date)`.  
2. Dataset sau xử lý có thể dùng để huấn luyện mô hình dự báo doanh thu.  

---

## 3. Kết quả mong đợi
- Dataset `orders` đã được chuẩn hóa (tên cột rõ ràng, kiểu dữ liệu đúng).  
- Có thể trực quan hóa và phân tích dữ liệu dễ dàng.  
- Flow trong Dataiku thể hiện đầy đủ quá trình **Upload → Explore → Prepare → Charts/Statistics**.  

---

## 4. Ghi chú
- File `orders.csv` phải được lưu trữ sẵn trên máy trước khi upload.  
- Nếu file có nhiều dữ liệu, nên kiểm tra hiệu năng khi hiển thị trong Dataiku.  
- Có thể xuất lại dataset sau xử lý thành file CSV mới bằng chức năng **Export**.  
"""
