import math


class DataPipeline:
    """Lớp DataPipeline tự triển khai từ đầu để thực hiện tiền xử lý dữ liệu.

    Các bước bao gồm: Xử lý missing values, Winsorization (xử lý ngoại lai),
    Chuẩn hóa Z-score, One-hot Encoding và kiểm tra Đa cộng tuyến (VIF).
    """

    def __init__(self, categorical_indices=None):
        # Danh sách chỉ mục các cột chứa biến phân loại
        self.cat_indices = categorical_indices if categorical_indices else []
        # Danh sách chỉ mục các cột chứa biến định lượng
        self.num_indices = []

        # Các dictionary để lưu trữ tham số học được từ tập Train
        self.impute_params = {}  # Lưu median và mode
        self.winsorize_bounds = {}  # Lưu cận trên và cận dưới
        self.scale_params = {}  # Lưu mean và std để chuẩn hóa Z-score
        self.onehot_categories = {}  # Lưu các danh mục duy nhất để làm One-hot Encoding

    def _is_missing(self, val):
        """Kiểm tra xem một giá trị có phải là giá trị khuyết thiếu hay không.

        Hỗ trợ nhận diện: None, math.isnan, chuỗi rỗng, chuỗi 'None', chuỗi
        'NaN'.
        """
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True

        # Nếu giá trị là chuỗi ký tự (str)
        if isinstance(val, str):
            val_clean = val.strip()
            # Kiểm tra chuỗi rỗng hoặc chuỗi chữ mang nội dung 'none', 'nan', 'null'
            if (
                val_clean == ""
                or val_clean.lower() == "none"
                or val_clean.lower() == "nan"
                or val_clean.lower() == "null"
            ):
                return True

        return False

    def _median(self, X):
        """Tính giá trị trung vị (Median) của một mảng số thực."""
        if not X:
            return 0.0
        valid = sorted(X)
        n = len(valid)
        mid = n // 2
        # Ép kiểu dữ liệu chắc chắn là float để thực hiện phép toán chia
        if n % 2 == 0:
            return (float(valid[mid - 1]) + float(valid[mid])) / 2.0
        return float(valid[mid])

    def _mode(self, X):
        """Tính giá trị yếu vị (Mode) của một mảng phân loại."""
        if not X:
            return None
        counts = {}
        for val in X:
            counts[val] = counts.get(val, 0) + 1
        max_count = -1
        mode_val = None
        for val, count in counts.items():
            if count > max_count:
                max_count = count
                mode_val = val
        return mode_val

    def fit(self, X):
        """Học các tham số tiền xử lý từ dữ liệu huấn luyện X (Train Set).

        X là cấu trúc dữ liệu dạng bảng 2 chiều (List of Lists).
        """
        if hasattr(X, "tolist"):
            X = X.tolist()
        elif hasattr(X, "values") and hasattr(X.values, "tolist"):
            X = X.values.tolist()
            
        # Kiểm tra rỗng an toàn cho Python List
        if len(X) == 0 or len(X[0]) == 0:
            return self

        n_rows = len(X)
        n_cols = len(X[0])

        # Phân loại danh sách chỉ mục các biến định lượng (numerical columns)
        self.num_indices = [
            j for j in range(n_cols) if j not in self.cat_indices
        ]

        # LẶP QUA TỪNG CỘT ĐỂ TÍNH THAM SỐ ĐIỀN KHUYẾT (IMPUTATION PARAMS)
        for j in range(n_cols):
            if j in self.cat_indices:
                # Cột biến phân loại: Lọc dữ liệu hợp lệ và tìm giá trị xuất hiện nhiều nhất (Mode)
                col_data = [
                    row[j] for row in X if not self._is_missing(row[j])
                ]
                self.impute_params[j] = self._mode(col_data)
            else:
                # Cột biến định lượng: Lọc dữ liệu, BẮT BUỘC ÉP KIỂU SANG FLOAT để xử lý
                col_data = []
                for row in X:
                    val = row[j]
                    if not self._is_missing(val):
                        try:
                            col_data.append(float(val))
                        except (ValueError, TypeError):
                            # Nếu dính lỗi định dạng chuỗi không thể ép kiểu (ví dụ dính ký tự chữ)
                            pass
                self.impute_params[j] = self._median(col_data)

        # Học tham số làm sạch nhiễu (Winsorization) và chuẩn hóa (Scaling) cho cột số
        # Tạo một bản sao dữ liệu đã điền khuyết tạm thời để tính toán phân vị, mean, std
        X_imputed = []
        for row in X:
            new_row = list(row)
            for j in range(n_cols):
                if self._is_missing(new_row[j]):
                    new_row[j] = self.impute_params[j]
                if j in self.num_indices:
                    new_row[j] = float(new_row[j])  # Đảm bảo chuyển hẳn thành float số
            X_imputed.append(new_row)

        # Tính toán biên ngoại lai bằng phương pháp IQR (Winsorization)
        for j in self.num_indices:
            col_data = sorted([row[j] for row in X_imputed])
            n = len(col_data)
            # Tính toán phân vị Q1 (25%) và Q3 (75%) thủ công
            q1 = col_data[int(n * 0.25)]
            q3 = col_data[int(n * 0.75)]
            iqr = q3 - q1
            self.winsorize_bounds[j] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Tính toán Mean và Standard Deviation (độ lệch chuẩn) phục vụ Z-score scaling
        for j in self.num_indices:
            col_data = [row[j] for row in X_imputed]
            # Giới hạn giá trị nằm trong khoảng Winsorize trước khi tính toán tham số chuẩn hóa
            lower, upper = self.winsorize_bounds[j]
            col_data_clipped = [
                min(max(val, lower), upper) for val in col_data
            ]

            mean_j = sum(col_data_clipped) / n_rows
            variance_j = (
                sum((val - mean_j) ** 2 for val in col_data_clipped) / n_rows
            )
            std_j = math.sqrt(variance_j)
            self.scale_params[j] = (mean_j, std_j if std_j != 0 else 1.0)

        # Học danh sách các nhãn phân loại duy nhất cho One-hot Encoding
        for j in self.cat_indices:
            unique_categories = sorted(
                list(set(str(row[j]) for row in X_imputed))
            )
            self.onehot_categories[j] = unique_categories

        return self

    def transform(self, X):
        """Áp dụng các tham số đã học được từ hàm fit để chuyển đổi ma trận dữ

        liệu X.

        Đầu ra trả về một ma trận số thực hoàn chỉnh (gồm cả cột Intercept 1.0
        ở vị trí đầu tiên).
        """
        if hasattr(X, "tolist"):
            X = X.tolist()
        elif hasattr(X, "values") and hasattr(X.values, "tolist"):
            X = X.values.tolist()

        if len(X) == 0 or len(X[0]) == 0:
            return []
        
        X_transformed = []
        for row in X:
            # 1. Điền giá trị khuyết thiếu (Imputation)
            imputed_row = list(row)
            for j in range(len(imputed_row)):
                if self._is_missing(imputed_row[j]):
                    imputed_row[j] = self.impute_params[j]

            # 2. Xử lý cột số (Winsorize + Z-score Standardization)
            processed_num_features = []
            for j in self.num_indices:
                val = float(imputed_row[j])  # Ép sang float
                lower, upper = self.winsorize_bounds[j]
                val_clipped = min(max(val, lower), upper)  # Winsorize

                mean_j, std_j = self.scale_params[j]
                val_scaled = (val_clipped - mean_j) / std_j  # Z-score
                processed_num_features.append(val_scaled)

            # 3. Xử lý cột chữ/phân loại (One-hot Encoding)
            processed_cat_features = []
            for j in self.cat_indices:
                val_str = str(imputed_row[j])
                categories = self.onehot_categories[j]
                # Tạo vector nhị phân 0-1 tương ứng cho từng nhãn danh mục
                for cat in categories:
                    processed_cat_features.append(
                        1.0 if val_str == cat else 0.0
                    )

            # 4. Gắn cột Intercept bằng 1.0 vào vị trí đầu tiên cùng các đặc trưng mới
            new_features = (
                [1.0] + processed_num_features + processed_cat_features
            )
            X_transformed.append(new_features)

        return X_transformed

    def calculate_vif(self, X_processed):
        """Tính chỉ số phóng đại phương sai (VIF) dựa trên ma trận dữ liệu đã xử

        lý.

        Bỏ qua cột đầu tiên (cột Intercept).
        """
        X = [row[1:] for row in X_processed]
        if not X or not X[0]:
            return []

        n_rows = len(X)
        n_cols = len(X[0])

        # 1: Tính toán ma trận hệ số tương quan tuyến tính Pearson (Correlation Matrix R)
        means = [sum(X[i][j] for i in range(n_rows)) / n_rows for j in range(n_cols)]
        stds = []
        for j in range(n_cols):
            variance = sum((X[i][j] - means[j]) ** 2 for i in range(n_rows)) / n_rows
            stds.append(math.sqrt(variance))

        R = [[0.0] * n_cols for _ in range(n_cols)]
        for j in range(n_cols):
            for k in range(n_cols):
                if stds[j] == 0 or stds[k] == 0:
                    R[j][k] = 0.0
                    continue
                cov = sum((X[i][j] - means[j]) * (X[i][k] - means[k]) for i in range(n_rows)) / n_rows
                R[j][k] = cov / (stds[j] * stds[k])

        # 2: Tìm ma trận nghịch đảo của R bằng phương pháp khử Gauss-Jordan
        aug = [R[i][:] + [1.0 if i == j else 0.0 for j in range(n_cols)] for i in range(n_cols)]

        for i in range(n_cols):
            pivot = aug[i][i]
            if abs(pivot) < 1e-8:
                return "Cảnh báo: Tồn tại đa cộng tuyến hoàn hảo, ma trận tương quan bị suy biến!"

            for j in range(2 * n_cols):
                aug[i][j] /= pivot

            for k in range(n_cols):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n_cols):
                        aug[k][j] -= factor * aug[i][j]

        # Trích xuất các phần tử nằm trên đường chéo chính của ma trận nghịch đảo R^-1 để lấy chỉ số VIF
        vif_values = [aug[i][n_cols + i] for i in range(n_cols)]
        return vif_values