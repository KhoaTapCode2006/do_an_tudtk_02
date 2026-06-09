import numpy as np
from ols_implementation import ols_fit
# Dùng thay thế cho hàm ols_fit(X, y)

# Lấy giá trị R^2 thay thế cho hàm model_metrics(y, y_hat, p)
def _r2_score_internal(y_true, y_pred):
    """
    Tính toán hệ số xác định R^2 (Coefficient of Determination).
    R2 = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def kfold_cv(X, y, k=5, random_state=42):
    """
    Thực hiện thuật toán k-Fold Cross-Validation.
    """
    X = np.asarray(X)
    y = np.asarray(y).flatten()
    n = len(y)
    
    # Tạo bộ sinh số ngẫu nhiên và xáo trộn chỉ số dữ liệu gốc (Shuffle)
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    # Chia các chỉ số đã xáo trộn thành k phần (folds) tương đương nhau
    folds = np.array_split(indices, k)
    scores = []
    
    # Vòng lặp huấn luyện và đánh giá mô hình xuyên suốt k fold
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.setdiff1d(indices, test_idx)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        try:
            # Huấn luyện và dự đoán
            beta_hat_2d, sigma2_hat = ols_fit(X_train, y_train)
            beta_hat = beta_hat_2d.flatten()

            y_hat_val = (X_test @ beta_hat).flatten()
            
            # Tính R2 gốc
            score = _r2_score_internal(y_test, y_hat_val)
            scores.append(score)
        except np.linalg.LinAlgError:
            scores.append(0.0)
            
    # Tính toán các thống kê tổng hợp (Trung bình và Độ lệch chuẩn thực tế)
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    
    # Làm tròn danh sách điểm số về 2 chữ số thập phân
    rounded_scores = [round(float(s), 2) for s in scores]
    
    # Định dạng chuỗi hiển thị
    mean_summary = f"Mean R2: {mean_r2:.3f} (+/- {std_r2:.2f})"
    
    # Xuất trực tiếp kết quả ra Console
    print(f"Scores: {rounded_scores}")
    print(mean_summary)
    
    return rounded_scores, mean_summary

# UNIT TESTS

def _test_cross_validation_implementation():
    print("=" * 60)
    print("RUNNING INTEGRITY TESTS FOR CROSS VALIDATION MODULE")
    print("=" * 60)
    
    # Unit Test 1: Dữ liệu ít nhiễu – mô hình khớp rất tốt
    print("[Unit Test 1] Dữ liệu tuyến tính ít nhiễu (scale=0.5)")
    rng = np.random.default_rng(2026)
    n = 120
    X1 = np.column_stack([np.ones(n), rng.uniform(1, 10, size=(n, 2))])
    y1 = 5.0 + 2.0 * X1[:, 1] - 1.2 * X1[:, 2] + rng.normal(scale=0.5, size=n)
    
    scores1, summary1 = kfold_cv(X1, y1, k=5, random_state=42)
    
    assert len(scores1) == 5,          "Lỗi: Số fold trả về phải bằng k=5."
    assert isinstance(summary1, str),  "Lỗi: summary phải là chuỗi ký tự."
    assert np.mean(scores1) > 0.90,    "Lỗi: Mean R² phải > 0.90 với dữ liệu ít nhiễu."
    print("Hoàn thành test, kết quả đúng.")
    
    # Unit Test 2: Noise lớn – mô hình khó khớp hơn
    print("\n[Unit Test 2] Dữ liệu tuyến tính, noise lớn (scale=5.0)")
    rng2 = np.random.default_rng(99)
    n2 = 150
    # Đã bổ sung đầy đủ các dấu đóng ngoặc ], ) bị khuyết của bạn
    X2 = np.column_stack([np.ones(n2), rng2.uniform(1, 10, size=(n2, 3))])
    y2 = 10.0 + 0.5 * X2[:, 1] + 3.0 * X2[:, 2] - 2.5 * X2[:, 3] + rng2.normal(scale=5.0, size=n2)
    
    scores2, summary2 = kfold_cv(X2, y2, k=5, random_state=42)
    
    assert len(scores2) == 5,          "Lỗi: Số fold trả về phải bằng k=5 ở Test 2."
    assert np.mean(scores2) < 0.85,    "Lỗi: Mean R² phải thấp hơn đáng kể (< 0.80) do nhiễu quá mạnh."
    print("Hoàn thành test, kết quả đúng.")
    

if __name__ == "__main__":
    _test_cross_validation_implementation()