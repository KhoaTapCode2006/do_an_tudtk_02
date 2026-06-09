import numpy as np
from cross_validation import kfold_cv, ols_fit

def tinh_aic_bic(y, y_hat, k_parameters):
    """Tính toán chỉ số thông tin AIC, BIC từ phần dư thực tế"""
    n = len(y)
    residuals = y - y_hat
    rss = residuals @ residuals
    
    if n == 0 or rss == 0:
        return 0.0, 0.0
        
    # Công thức toán học chuẩn tắc của AIC và BIC cho hồi quy OLS
    aic = n * np.log(rss / n) + 2 * k_parameters
    bic = n * np.log(rss / n) + k_parameters * np.log(n)
    return round(aic, 2), round(bic, 2)

def compare_model(X_raw, y_raw):
    # Trích xuất cột biến độc lập chính x (bỏ qua cột intercept 1 nếu có)
    x = X_raw[:, 1] if len(X_raw.shape) > 1 and X_raw.shape[1] > 1 else X_raw
    x = x.flatten()
    
    # 1. Định hình ma trận thiết kế (Design Matrix) cho 3 mô hình đề xuất
    X_tuyen_tinh = np.column_stack([np.ones_like(x), x])
    X_bac2       = np.column_stack([np.ones_like(x), x, x**2])
    X_overfit    = np.column_stack([np.ones_like(x)] + [x**i for i in range(1, 10)]) # Bậc 9

    scores_tt, summary_tt = kfold_cv(X_tuyen_tinh, y_raw, k=5, random_state=42)
    scores_b2, summary_b2 = kfold_cv(X_bac2, y_raw, k=5, random_state=42)
    scores_ov, summary_ov = kfold_cv(X_overfit, y_raw, k=5, random_state=42)

    # 3. Tính toán nhanh chuỗi giá trị y_hat để lấy AIC/BIC
    _, y_hat_tt = ols_fit(X_tuyen_tinh, y_raw)
    _, y_hat_b2 = ols_fit(X_bac2, y_raw)
    _, y_hat_ov = ols_fit(X_overfit, y_raw) 
    
    aic_tt, bic_tt = tinh_aic_bic(y_raw, y_hat_tt, X_tuyen_tinh.shape[1])
    aic_b2, bic_b2 = tinh_aic_bic(y_raw, y_hat_b2, X_bac2.shape[1])
    aic_ov, bic_ov = tinh_aic_bic(y_raw, y_hat_ov, X_overfit.shape[1])

    # 4. Gom dữ liệu thiết lập cấu trúc Dictionary
    models = [
        "Mo hinh tuyen tinh don gian",
        "Mo hinh da thuc bac 2 (Toi uu)",
        "Mo hinh Overfitting (Bac cao)"
    ]
    
    # Lấy chuỗi thống kê "Mean R2: ... (+/- ...)"
    mean_std_str = [
        summary_tt.split(":", 1)[1].strip(),
        summary_b2.split(":", 1)[1].strip(),
        summary_ov.split(":", 1)[1].strip()
    ]
    
    data = {
        "Cau truc Mo hinh": models,
        "Scores tung Fold (R2)": [str(scores_tt), str(scores_b2), str(scores_ov)],
        "Hieu nang Mean R2 (+/- STD)": mean_std_str,
        "AIC": [aic_tt, aic_b2, aic_ov],
        "BIC": [bic_tt, bic_b2, bic_ov]
    }
    
    data = {
        "Cau truc Mo hinh": models,
        "Scores tung Fold (R2)": [str(scores_tt), str(scores_b2), str(scores_ov)],
        "Hieu nang Mean R2 (+/- STD)": mean_std_str,
        "AIC": [aic_tt, aic_b2, aic_ov],
        "BIC": [bic_tt, bic_b2, bic_ov]
    }
    
    w_model = 32
    w_folds = 36
    w_mean  = 24
    w_aic   = 10
    w_bic   = 10
    
    sep_line = f"+{'-'*(w_model+4)}+{'-'*(w_folds+4)}+{'-'*(w_mean+4)}+{'-'*(w_aic+4)}+{'-'*(w_bic+4)}+"
    
    print("\n" + "=" * 30 + " BANG DOI CHIEU HIEU NANG MO HINH HOI QUY " + "=" * 30)
    print(sep_line)
    print(f"|  {'Cau truc Mo hinh'.ljust(w_model)}  |  {'Scores tung Fold (R2)'.center(w_folds)}  |  {'Mean R2 (+/- STD)'.center(w_mean)}  |  {'AIC'.center(w_aic)}  |  {'BIC'.center(w_bic)}  |")
    print(sep_line)
    
    for i in range(3):
        c_model = data["Cau truc Mo hinh"][i].ljust(w_model)
        c_folds = data["Scores tung Fold (R2)"][i].center(w_folds)
        c_mean  = data["Hieu nang Mean R2 (+/- STD)"][i].center(w_mean)
        c_aic   = f"{data['AIC'][i]:.2f}".center(w_aic)
        c_bic   = f"{data['BIC'][i]:.2f}".center(w_bic)
        print(f"|  {c_model}  |  {c_folds}  |  {c_mean}  |  {c_aic}  |  {c_bic}  |")
        
    print(sep_line)
    print("=" * 80 + "\n")

if __name__ == "__main__":
    # Giả lập 120 mẫu dữ liệu thực nghiệm (có cấu trúc cong bậc 2)
    rng = np.random.default_rng(42)
    n_samples = 120
    x_test = rng.uniform(-2.5, 2.5, size=n_samples)
    y_test = 3.5 + 2.2 * x_test + 1.5 * (x_test**2) + rng.normal(scale=1.2, size=n_samples)
    
    # Thực thi hàm vẽ bảng
    compare_model(x_test, y_test)