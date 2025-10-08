## README – LAB4: Data Preparation và Visualization trong Dataiku

##  1. Làm việc với cột ngày tháng (Date columns)
- Khi cột chứa dữ liệu dạng `"MM/dd/yyyy"`, cần **parse** về kiểu ngày trước khi tính toán:
  - Dùng bước **“Parse date in column”** để tạo ra các cột như:
    - `Start Date_parsed`
    - `End Date_parsed`
- Sau khi parse, kiểu dữ liệu của cột sẽ là **datetime with tz** (có thể sử dụng được trong công thức).

---

##  2. Tạo cột mới bằng công thức (Formula)
- Dùng step **“Create column with formula”** trong phần **Prepare Recipe**.
- Có thể viết công thức trực tiếp trong ô *Expression*.

###  Ví dụ: Tính số ngày giữa hai mốc thời gian
diff("End Date_parsed", "Start Date_parsed", "days")

## `diff()` là hàm nội bộ của Dataiku, dùng để tính hiệu giữa hai giá trị thời gian.  
Tham số thứ 3 `"days"` xác định đơn vị kết quả.

---

##  3. Một số hàm thông dụng trong Dataiku Formula
| Hàm | Chức năng | Ví dụ |
|-----|------------|-------|
| `if(condition, value_if_true, value_if_false)` | Câu điều kiện | `if(score > 10, "Good", "Normal")` |
| `diff(date1, date2, "days")` | Hiệu giữa hai ngày | `diff("End Date", "Start Date", "days")` |
| `concat(str1, str2)` | Nối chuỗi | `concat("A", "B")` |
| `numval("Column")` | Chuyển cột sang dạng số | `numval("Score") + 5` |
| `strval("Column")` | Chuyển cột sang dạng chuỗi | `strval("State")` |

---

##  4. Vẽ biểu đồ (Charts)
Để trực quan hóa dữ liệu, mở dataset và chọn tab **Charts**.

###  Cấu hình Scatter Plot
- **Chart type:** Scatter Plot  
- **X-axis:** `Facility Name`  
- **Y-axis:** `Number of Patients` → Aggregation: `SUM`  
- **Color by (optional):** `State`  
- **Size by (optional):** `Score`  
- **Tooltip:** Thêm `Measure Name`, `Days` để hiển thị khi hover chuột.

 Kết quả: Biểu đồ thể hiện tổng số bệnh nhân theo từng cơ sở y tế (Facility Name).

---

##  5. Kinh nghiệm khắc phục lỗi thường gặp
| Lỗi | Nguyên nhân | Cách khắc phục |
|------|--------------|----------------|
| `Unknown function 'toDate'` | Dataiku không hỗ trợ `toDate()` | Dùng `Parse date step` để chuyển đổi trước |
| `Invalid format: "End Date_parsed"` | Cột chưa được parse đúng kiểu | Kiểm tra lại step Parse Date |
| `Cannot parse to number` | Dữ liệu đang ở dạng chuỗi | Dùng `numval()` để chuyển sang số |

---

##  6. Kết luận
- Khi thao tác với ngày tháng trong Dataiku, **luôn parse cột trước khi tính toán**.  
- Sử dụng hàm **diff()** để tính khoảng cách giữa hai ngày.  
- Dùng tab **Charts** để trực quan hóa dữ liệu nhanh chóng mà không cần code.  
- Dataiku cung cấp các hàm nội bộ rất mạnh — không cần dùng thư viện ngoài như Python.
"""



