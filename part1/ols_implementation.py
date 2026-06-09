import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import json
import math # Thêm thư viện math của Python chuẩn để dùng hàm sqrt
# Các hàm hỗ trợ
def transpose(A): # chuyển vị
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B): # nhân hai ma trận
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in transpose(B)] for A_row in A]

def invert_matrix(A):
    """Tính ma trận nghịch đảo của ma trận vuông A bằng phương pháp khử Gauss-Jordan

    kết hợp chọn phần tử trội (Partial Pivoting) và tự động sửa đổi đường chéo
    nếu suy biến.
    """
    n = len(A)

    # Sao chép ma trận để tránh thay đổi trực tiếp ma trận đầu vào
    A_copy = [row[:] for row in A]

    # Kiểm tra kiểm định an toàn: Nếu phần tử đường chéo chính quá nhỏ (suy biến),
    # chủ động cộng thêm một lượng nhiễu siêu nhỏ (Ridge Jitter / Tikhonov Regularization)
    # để ép ma trận trở nên khả nghịch, bảo vệ thuật toán Gauss-Jordan.
    is_singular = False
    for i in range(n):
        if abs(A_copy[i][i]) < 1e-9:
            is_singular = True
            break

    if is_singular:
        # Cộng thêm jitter lambda = 1e-7 vào đường chéo chính của X^T * X
        for i in range(n):
            A_copy[i][i] += 1e-7

    # Khởi tạo ma trận bổ sung [A | I]
    AM = [
        A_copy[i][:] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for i in range(n):
        # 1. Thuật toán chọn phần tử trội (Partial Pivoting): Tìm dòng từ i trở xuống
        # có giá trị tuyệt đối tại cột i lớn nhất để hoán vị lên đầu.
        pivot_row = i
        max_val = abs(AM[i][i])
        for j in range(i + 1, n):
            if abs(AM[j][i]) > max_val:
                max_val = abs(AM[j][i])
                pivot_row = j

        # Đổi chỗ dòng hiện tại với dòng chứa phần tử trội
        if pivot_row != i:
            AM[i], AM[pivot_row] = AM[pivot_row], AM[i]

        pivot = AM[i][i]

        # Phòng ngừa nếu định thức vẫn bằng 0
        if abs(pivot) < 1e-12:
            # Nếu pivot quá gần 0, gán tạm bằng epsilon để ngăn lỗi chia cho 0
            pivot = 1e-12 if pivot >= 0 else -1e-12

        # 2. Chuẩn hóa dòng hiện tại (Đưa phần tử chéo về 1)
        for j in range(2 * n):
            AM[i][j] /= pivot

        # 3. Khử Gauss để biến đổi tất cả các phần tử khác trên cột i về 0
        for j in range(n):
            if i != j:
                factor = AM[j][i]
                for k in range(2 * n):
                    AM[j][k] -= factor * AM[i][k]

    # Trích xuất ma trận bên phải chính là kết quả nghịch đảo A^-1
    return [row[n:] for row in AM]


# Các hàm hỗ trợ
def calculate_tss(y):
    # Tính giá trị trung bình của y
    y_bar = sum(y) / len(y)
    
    # Tính tổng bình phương các độ lệch (y_i - y_bar)^2
    return sum((val - y_bar) ** 2 for val in y)

def calculate_rss(y, y_hat):
    # Kiểm tra độ dài 2 list phải bằng nhau
    if len(y) != len(y_hat):
        raise ValueError("Độ dài của y và y_hat phải bằng nhau.")
        
    # Tính tổng bình phương các phần dư (y_i - y_hat_i)^2
    return sum((yi - yi_hat) ** 2 for yi, yi_hat in zip(y, y_hat))

def flatten(list_of_lists):
    """Chuyển đổi danh sách lồng nhau thành danh sách phẳng (1 chiều)."""
    result = []
    for item in list_of_lists:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))  # Đệ quy nếu gặp list con
        else:
            result.append(item)
    return result
# =====================================================================
# 1. HÀM OLS FITTING 
# =====================================================================

def ols_fit(X, y):
    X_list = np.asarray(X).tolist()
    y_list = np.asarray(y).tolist()

    if len(np.array(y_list).shape) == 1:
        y_list = [[val] for val in y_list]

    n = len(X_list)
    p = len(X_list[0])  # p là tổng số cột

    XT = transpose(X_list)
    XT_X = matmul(XT, X_list)
    XT_X_inv = invert_matrix(XT_X)
    XT_y = matmul(XT, y_list)

    beta_hat = matmul(XT_X_inv, XT_y)

    y_hat = matmul(X_list, beta_hat)
    rss = sum((y_list[i][0] - y_hat[i][0]) ** 2 for i in range(n))
    sigma2_hat = rss / (n - p)

    return np.array(beta_hat), float(sigma2_hat)


# =====================================================================
# 2. HÀM TÍNH HAT MATRIX 
# =====================================================================

def hat_matrix(X):
    X_list = np.asarray(X).tolist()
    XT = transpose(X_list)
    XT_X = matmul(XT, X_list)
    XT_X_inv = invert_matrix(XT_X)

    part1 = matmul(X_list, XT_X_inv)
    H = matmul(part1, XT)

    return np.array(H)
# unitest



# =====================================================================
# 3. HÀM ĐÁNH GIÁ MÔ HÌNH (MODEL METRICS)
# =====================================================================
def model_metrics(y, y_hat, p):
    """
    Đánh giá độ chính xác của mô hình hồi quy OLS.
    """

    y = flatten(y)
    y_hat = flatten(y_hat)
    
    n = len(y)
    
    TSS = calculate_tss(y)
    RSS = calculate_rss(y, y_hat)
    
    R2 = 1 - (RSS / TSS)
    Adj_R2 = 1 - ((n - 1) / (n - p - 1)) * (1 - R2)
    
    if np.isclose(RSS, 0.0):
        F_stat = np.inf
    else:
        F_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))
    
    return {
        "RSS": float(np.round(RSS, 4)),
        "TSS": float(np.round(TSS, 4)),
        "R2": float(np.round(R2, 4)),
        "Adj_R2": float(np.round(Adj_R2, 4)),
        "F_stat": float(np.round(F_stat, 4))
    }

# =====================================================================
# 3. HÀM SUY DIỄN HỆ SỐ (COEFFICIENT INFERENCE)
# =====================================================================

def coef_inference(X, y, beta_hat, sigma2_hat):
    """
    Thực hiện các suy diễn thống kê cho từng hệ số hồi quy thuần bằng Python Lists.
    """
    # 1. Chuyển đổi dữ liệu (có thể là mảng numpy từ main) sang dạng List 2 chiều thuần Python
    X_list = [list(row) for row in X]
    beta_list = [float(b) for b in beta_hat]
    sigma2_val = float(sigma2_hat)
    
    n = len(X_list)
    p_plus_1 = len(X_list[0])
    p = p_plus_1 - 1
    df = n - p - 1
    
    # 2. Tính ma trận chuyển vị X^T
    X_T = []
    for i in range(p_plus_1):
        X_T.append([X_list[j][i] for j in range(n)])
        
    # 3. Tính tích hai ma trận (X^T * X)
    XTX = []
    for i in range(p_plus_1):
        row = []
        for j in range(p_plus_1):
            # Nhân dòng i của X_T với cột j của X
            s = sum(X_T[i][k] * X_list[k][j] for k in range(n))
            row.append(s)
        XTX.append(row)
        
    # 4. Tìm ma trận nghịch đảo C = (X^T * X)^-1 bằng thuật toán Khử Gauss-Jordan
    # Tạo ma trận mở rộng [XTX | Ma trận đơn vị I]
    aug = []
    for i in range(p_plus_1):
        row = XTX[i][:] + [1.0 if i == j else 0.0 for j in range(p_plus_1)]
        aug.append(row)
        
    for i in range(p_plus_1):
        # Tìm phần tử trội để xoay vòng (Pivot) nhằm tránh chia cho 0
        pivot_row = i
        for k in range(i+1, p_plus_1):
            if abs(aug[k][i]) > abs(aug[pivot_row][i]):
                pivot_row = k
                
        # Hoán vị dòng chứa phần tử trội lên trên
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]
        
        pivot = aug[i][i]
        if pivot == 0:
            raise ValueError("Ma trận suy biến (Singular Matrix), không thể tìm nghịch đảo.")
            
        # Chia toàn bộ dòng i cho số pivot để đưa phần tử đường chéo về 1
        for j in range(2 * p_plus_1):
            aug[i][j] /= pivot
            
        # Biến các phần tử còn lại trên cột i thành 0 (Khử Gauss)
        for k in range(p_plus_1):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * p_plus_1):
                    aug[k][j] -= factor * aug[i][j]
                    
    # Tách lấy nửa bên phải của ma trận mở rộng chính là ma trận nghịch đảo C
    C = [row[p_plus_1:] for row in aug]
    
    # 5. Lấy đường chéo chính của ma trận C
    diag_C = [C[i][i] for i in range(p_plus_1)]
    
    # 6. Tính toán các chỉ số thống kê bằng logic Python cơ bản
    se_beta = [math.sqrt(sigma2_val * d) for d in diag_C]
    t_stat = [b / se for b, se in zip(beta_list, se_beta)]
    
    # Gọi scipy.stats để lấy giá trị kiểm định
    p_values = [2 * stats.t.sf(abs(t), df=df) for t in t_stat]
    t_crit = stats.t.ppf(0.975, df=df) 
    
    ci_lower = [b - t_crit * se for b, se in zip(beta_list, se_beta)]
    ci_upper = [b + t_crit * se for b, se in zip(beta_list, se_beta)]
    
    # 7. Đóng gói kết quả
    var_names = ['Intercept'] + [f'X{i}' for i in range(1, p_plus_1)]
    df_inference = pd.DataFrame({
        'Biến': var_names,
        'Hệ số': beta_list,
        'Sai số chuẩn': se_beta,
        't-stat': t_stat,
        'p-value': p_values,
        '[0.025]': ci_lower,
        '[0.975]': ci_upper
    })
    
    cols_to_round = ['Hệ số', 'Sai số chuẩn', 't-stat', 'p-value', '[0.025]', '[0.975]']
    df_inference[cols_to_round] = df_inference[cols_to_round].round(4)
    
    return df_inference


def run_tests_ols_hat():
    print("=" * 50)
    print("Hàm ols_fit(X, y)")
    print("=" * 50)

    for i in range(1, 3):
        print(f"\n--- Test Case {i} ---")
        np.random.seed(42)
        n, p = 20 * i, 3 * i

        # Giả lập dữ liệu
        X_features = np.random.randn(n, p)
        X = np.c_[np.ones(n), X_features]  # Thêm 1 cột intercept
        true_beta = np.random.randn(p + 1, 1)
        y = X.dot(true_beta) + np.random.randn(n, 1) * 0.5

        # 1. Kết quả từ ols_fit
        beta_hat, sigma2_hat = ols_fit(X, y)

        # 2. Kết quả từ sklearn
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        beta_sklearn = model.coef_.T

        y_pred = model.predict(X)
        rss_sk = np.sum((y - y_pred) ** 2)
        sigma2_sklearn = rss_sk / (n - (p + 1))

        print(f"Beta Match (Custom vs Sklearn): {np.allclose(beta_hat, beta_sklearn)}")
        print(
            f"Sigma2 Match: Custom = {sigma2_hat:.5f}, Sklearn = {sigma2_sklearn:.5f} -> {np.isclose(sigma2_hat, sigma2_sklearn)}")

    print("\n" + "=" * 50)
    print("Hàm hat_matrix(X)")
    print("=" * 50)

    for i in range(1, 3):
        print(f"\n--- Test Case {i} ---")
        np.random.seed(42)
        n, p = 15 * i, 2 * i
        X_features = np.random.randn(n, p)
        X = np.c_[np.ones(n), X_features]

        # 1. Tính Hat Matrix
        H = hat_matrix(X)
        H_numpy = X @ np.linalg.inv(X.T @ X) @ X.T

        # 2. Kiểm chứng với công thức NumPy
        print(f"H match công thức Numpy: {np.allclose(H, H_numpy)}")

        # 3. Kiểm tra tính Đối xứng (H = H^T)
        print(f"Kiểm tra Đối xứng (Symmetric): {np.allclose(H, H.T)}")

        # 4. Kiểm tra tính Lũy đẳng (H^2 = H) - Idempotent
        is_idempotent = np.allclose(H @ H, H)
        print(f"Kiểm tra Lũy đẳng (Idempotent): {is_idempotent}")
        
def run_tests_model_metrics():
    print("Tạo dữ liệu giả lập cho model metrics và suy diễn hệ số")
    np.random.seed(42)
    
    n = 100
    p = 4
    X_features = np.random.randn(n, p)
    X = np.c_[np.ones(n), X_features]
    
    true_beta = np.array([5.02, 3.48, -2.0, 1.5, 0.0])
    noise = np.random.randn(n) * 0.5
    y = X @ true_beta + noise
    
    beta_hat_np = np.linalg.pinv(X.T @ X) @ X.T @ y
    y_hat_np = X @ beta_hat_np
    RSS_np = np.sum((y - y_hat_np)**2)
    sigma2_hat_np = RSS_np / (n - p - 1)

    print("Kiểm thử hàm model_metrics")
    reg = LinearRegression().fit(X_features, y)
    y_hat_sklearn = reg.predict(X_features)
    
    metrics = model_metrics(y, y_hat_sklearn, p)
    
    assert np.isclose(metrics['R2'], np.round(reg.score(X_features, y), 4)), "Lỗi: R2 không khớp sklearn!"
    perfect_metrics = model_metrics(y, y, p)
    assert np.isclose(perfect_metrics['R2'], 1.0), "Lỗi: R2 của Perfect Fit phải bằng 1"
    print("model_metrics: PASSED 2/2 TESTS! ")
    
    print("\n" + "="*75)
    print("TỪ ĐIỂN CHỈ SỐ MÔ HÌNH (MODEL METRICS OUTPUT):")
    print("="*75)
    print(json.dumps(metrics, indent=4))
    print("="*75)

    print("Kiểm thử hàm coef_inference")
    df_result = coef_inference(X, y, beta_hat_np, sigma2_hat_np)
    
    assert np.allclose(df_result['Sai số chuẩn'].values, np.round(expected_se:=np.sqrt(sigma2_hat_np * np.diag(np.linalg.pinv(X.T @ X))), 4)), "Lỗi: Sai số chuẩn sai!"
    
    lower_bound = df_result['[0.025]'].values
    upper_bound = df_result['[0.975]'].values
    actual_beta = df_result['Hệ số'].values
    diff_lower = np.round(actual_beta - lower_bound, 4)
    diff_upper = np.round(upper_bound - actual_beta, 4)
    assert np.allclose(diff_lower, diff_upper, atol=1e-3), "Lỗi: Khoảng tin cậy không đối xứng!"
    print("coef_inference: PASSED 2/2 TESTS!")
    
    print("\n" + "="*75)
    print("BẢNG KẾT QUẢ SUY DIỄN HỆ SỐ:")
    print("="*75)
    print(df_result.to_string(index=False))
    print("="*75)

if __name__ == "__main__":
    run_tests_ols_hat()
    run_tests_model_metrics()