# Đồ Án Toán Ứng Dụng và Thống Kê 02

## 📌 Tổng quan dự án
Đồ án tập trung vào các việc sau đây:

* **Nghiên cứu và chứng minh toán học:** Hệ thống hóa cơ sở lý thuyết của bài toán ước lượng tối ưu bằng phương pháp bình phương tối thiểu (OLS), các giả thiết hệ quả Gauss-Markov, và bản chất hình học của không gian phần dư.
* **Xây dựng DataPipeline và huấn luyện:** Phát triển quy trình tiền xử lý dữ liệu thô toàn diện (làm sạch, mã hóa, chuẩn hóa) và cài đặt các thuật toán hồi quy tuyến tính (OLS Cơ bản, OLS Chọn biến) phối hợp cùng hồi quy điều hòa (Ridge) qua cơ chế kiểm định chéo $K\text{-Fold}$.
* **Đánh giá và phân tích thực nghiệm:** Xây dựng hệ thống đồ thị chẩn đoán sai số chuyên sâu, tính toán đối chiếu các chỉ số lỗi định lượng nhằm tối ưu hóa bài toán thực tế dự báo cước phí vận tải đô thị.

* **Trường:** Đại học Khoa học Tự nhiên - ĐHQG TP.HCM (HCMUS)
* **Học phần:** Toán ứng dụng và thống kê 
* **Lớp:** 24CTT2
* **Ngôn ngữ:** Python 3.10+
* **Công cụ hỗ trợ:** LaTeX, Jupyter Notebook, Manim.
* **Giảng viên thực hành:**
    * ThS. Võ Nam Thục Đoan
    * ThS. Lê Nhựt Nam
    
---

## Sinh viên thực hiện
| MSSV     | Họ và Tên              | 
| :---     | :---                   | 
| 24120075 | Trần Nguyễn Anh Khoa   | 
| 24120048 | Lê Hoàng Hiếu          | 
| 24120131 | Nguyễn Anh Sang        | 
| 24120061 | Lê Nguyễn Gia Huy      | 
| 24120097 | Nguyễn Hoàng Nam       | 


---

## Tổng quan các phần của đồ án

### Phần 1: Lý thuyết Data Fitting và Minh họa Hình học
* **Giả thiết Gauss-Markov:** Nghiên cứu và biện giải hệ thống các giả thiết cốt lõi để chứng minh ước lượng OLS đạt tính chất BLUE (Ước lượng tuyến tính không chệch tốt nhất).
* **Bình phương tối thiểu:** Thiết lập công thức giải tích chuẩn tắc hệ số hồi quy dưới dạng phương trình ma trận:
  $$\hat{\beta} = (X^T X)^{-1} X^T y$$
* **Đa cộng tuyến và Kiểm định VIF:** Nghiên cứu hiện tượng phụ thuộc tuyến tính giữa các biến độc lập và cài đặt chỉ số VIF (Variance Inflation Factor) để đo lường mức độ phóng đại phương sai của hệ số.
* **K-Fold Cross Validation:** Xây dựng cơ chế chia tách tập dữ liệu huấn luyện thành $K$ phần độc lập nhằm đánh giá tính tổng quát hóa của mô hình và tự động dò tìm siêu tham số điều hòa $\lambda$ tối ưu.
* **Mô phỏng Monte Carlo:** Trình bày lý thuyết và các bước thực hiện Monte Carlo, minh hoạ Monte Carlo với dữ liệu giả lập và vẽ biểu đồ Histogram, Q-Q, Boxplots, Bias để kiểm chứng tính không chệch (OLS là BLUE)

### Phần 2: Xây dựng DataPipeline, Huấn luyện Mô hình, Phân tích Phần dư và Đánh giá Hiệu năng Hệ thống
* **Tiền xử lý dữ liệu:** Thực hiện EDA, xử lý missing values loại MAR (Bằng phương pháp Điền khuyết bằng đại lượng thống kê trung tâm), khử nhiễu/Outliers bằng phương pháp Winsorization, thực hiện mã hóa một lựa chọn (One-Hot Encoding) biến phân loại và chuẩn hóa thang đo đặc trưng (Z-score).
* **Mô hình OLS và chọn biến:** Cài đặt cấu trúc hồi quy đa biến tuyến tính thuần túy; đồng thời tích hợp thuật toán lọc biến tự động dựa trên mức độ ý nghĩa thống kê ($p\text{-value}$) và kiểm soát đa cộng tuyến (VIF).
* **Điều hòa Ridge:** Cài đặt phương pháp tối ưu hóa có ràng buộc để thu hẹp hệ số hồi quy nhằm chống hiện tượng quá khớp (Overfitting), sử dụng hình phạt cấu trúc hệ số dạng $L_2$ (Ridge).
* **Hệ đồ thị chẩn đoán (Diagnostic Plots):** Triển khai xây dựng và phân tích bộ 4 biểu đồ thống kê tiêu chuẩn bao gồm: *Residuals vs Fitted* (kiểm tra tính tuyến tính), *Normal Q-Q* (kiểm tra phân phối chuẩn của sai số), *Scale-Location* (kiểm tra phương sai đồng đều), và *Residuals vs Leverage* (phát hiện điểm dữ liệu có sức ảnh hưởng cực đoan).
* **Chỉ số đo lường hiệu năng:** Tính toán, lập bảng đối chiếu kết quả thực nghiệm giữa các mô hình trên tập kiểm thử (Test Set) thông qua ba đại lượng cốt lõi: Sai số tuyệt đối trung bình (MAE), Căn trung bình bình phương sai số (RMSE), và Hệ số xác định ($R^2$).
* **Biện giải thực tế ứng dụng:** Kết hợp kết quả số học của mô hình với bản chất hành vi kinh tế từ bộ dữ liệu thực tế (Taxi Yellow NYC) nhằm đưa ra kết luận về tính hợp lý của cước phí, cấu trúc khách hàng, và đề xuất giải pháp ứng dụng tính giá cước thời gian thực.

---

## Cấu trúc thư mục 
```text
.
├── img/                        # Thư mục chứa hình ảnh các biểu đồ được vẽ từ part1 và part2
├── reports/
│   └── reports.pdf             # File report PDF
├── part1/
│   ├── compare_models.py       # So sánh 3 mô hình tuyến tính, bậc 2 và overfitting
│   ├── cross_validation.py     # Thuật toán chia K-Folds và tính R^2
│   ├── ols_implementation.py   # Hàm mất mát và tìm nghiệm OLS, tính ma trận Hat, đánh giá mô hình và thực hiện suy diễn thống kê  
│   ├── residual_analysis.py    # Vẽ 4 biểu đồ chẩn đoán phần dư để kiểm tra sai số cho mô hình
│   ├── ridge_lasso.py          # Tính hệ số VIF kiểm tra đa cộng tuyến và cài đặt thành phần chính quy hoá Ridge
│   └── part1_notebook.ipynb    # File Jupyter Notebook minh hoạ Monte Carlo (Bao gồm vẽ biểu đồ)
├── part2/
│   ├── data/
|   │   └── yellow_tripdata_2026-02.csv      # Bộ dữ liệu CSV được sử dụng
│   ├── eda_plots/              # Thư mục chứa các file biểu đồ EDA từ bộ dữ liệu CSV
│   ├── fit_and_transform.py    # Thực hiện chuẩn hoá dữ liệu và xử lí missing values 
│   ├── models_wrapper.py       # Cài đặt 3 mô hình trong class Models
│   ├── model_comparison.py     # Huấn luyện 3 mô hình và đánh giá mô hình dựa trên tập test (Bộ dữ liệu CSV), vẽ 4 biểu đồ chẩn đoán phần dư
│   └── part2_notebook.ipynb    # File Jupyter Notebook đánh giá mô hình và phân tích chuyên sâu bộ dữ liệu
|
├── requirements.txt    # File requirements chứa các thư viện cần thiết
└── README.md           # Markdown chứa tổng quan và hướng dẫn cài thư viện
```

## Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
* Python 3.10 trở lên.
* Trình quản lý gói `pip`.

### 2. Tạo môi trường ảo Venv
Mở terminal và chạy lệnh: 
# Windows
+ python -m venv .venv
# Linux / macOS
+ python3 -m venv .venv
+ source .venv/bin/activate
# Ubuntu
+ sudo apt-get install python3-venv
+ python3 -m venv .venv
+ source .venv/bin/activate

### 3. Cài đặt thư viện phụ thuộc
Tất cả các thư viện cần thiết cho đồ án đều được lưu trữ trong file requirements.txt, mở terminal và chạy lệnh sau
+ pip install -r requirements.txt
