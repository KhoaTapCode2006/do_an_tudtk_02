"""
===================================================
Đánh giá và so sánh 3 mô hình hồi quy trên tập Test.

Quy trình:
    1. Đọc dữ liệu thô, tách Train/Test (80/20, seed=42)
    2. Fit DataPipeline trên Train, transform cả Train lẫn Test
    3. Nhận beta từ 3 mô hình 
    4. evaluate_models() → DataFrame MAE / RMSE / R²  
    5. Vẽ 4 biểu đồ chẩn đoán cho mô hình tốt nhất
"""

import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))  
main_dir = os.path.dirname(current_dir)  
part1_dir = os.path.join(main_dir, "part1")  


if main_dir not in sys.path:
    sys.path.append(main_dir)
if part1_dir not in sys.path:
    sys.path.append(part1_dir)

import math
import random
import pandas as pd
import numpy as np


from fit_and_transform import DataPipeline
from models_wrapper import Models
from part1.residual_analysis import residual_plots # Tái sử dụng hàm residual_plots() của part1


# PHẦN 1 — HÀM TÍNH METRICS
def _dot(row, beta_flat):
    """Nhân một hàng vector với beta (1D list)."""
    return sum(x * b for x, b in zip(row, beta_flat))


def _predict(X_processed, beta):
    """Tính y_hat = X_processed · beta bằng phép nhân ma trận."""
    # Làm phẳng beta về dạng 1D để nhân ma trận
    if beta and isinstance(beta[0], list):
        beta_flat = [b[0] for b in beta]
    else:
        beta_flat = list(beta)

    return [_dot(row, beta_flat) for row in X_processed]


def _mae(y_true, y_pred):
    """Mean Absolute Error."""
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def _rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    n = len(y_true)
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n
    return math.sqrt(mse)


def _r2(y_true, y_pred):
    """Hệ số xác định R² trên tập Test: R²_test = 1 - RSS_test / TSS_test"""
    n = len(y_true)
    y_mean = sum(y_true) / n
    rss = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    tss = sum((yt - y_mean) ** 2 for yt in y_true)
    if tss == 0:
        return 0.0
    return 1.0 - rss / tss


# PHẦN 2 — HÀM ĐÁNH GIÁ CHÍNH
def evaluate_models(
    X_test_processed,
    y_test_raw,
    beta_ols_full,
    beta_ols_selective,
    selected_indices,
    beta_regularized,
    model_type_label="Ridge",
):
    """So sánh 3 mô hình hồi quy trên tập Test và trả về DataFrame kết quả."""

    # ── Bước 1: Chuẩn hóa y_test về dạng list 1D ────────────────────────────
    # X_test_processed đã được transform từ bên ngoài (trong main())
    if hasattr(y_test_raw, "tolist"):
        y_test = y_test_raw.tolist()
    elif hasattr(y_test_raw, "values"):
        y_test = y_test_raw.values.tolist()
    else:
        y_test = list(y_test_raw)

    # Làm phẳng nếu là list 2D
    if y_test and isinstance(y_test[0], list):
        y_test = [v[0] for v in y_test]
    y_test = [float(v) for v in y_test]

    # ── Bước 3: Dự đoán cho từng mô hình ────────────────────────────────────

    # Mô hình 1: OLS Cơ bản — dùng toàn bộ X_test_processed
    y_hat_full = _predict(X_test_processed, beta_ols_full)

    # Mô hình 2: OLS Chọn biến — chỉ lấy các cột trong selected_indices
    # (Cần lọc X_test_processed theo đúng các cột đã chọn lúc train)
    X_test_selective = [
        [row[idx] for idx in selected_indices]
        for row in X_test_processed
    ]
    y_hat_selective = _predict(X_test_selective, beta_ols_selective)

    # Mô hình 3: Ridge — dùng toàn bộ X_test_processed
    y_hat_reg = _predict(X_test_processed, beta_regularized)

    # ── Bước 4: Tính 3 chỉ số đánh giá cho mỗi mô hình ─────────────────────
    results = {
        "Mô hình": ["OLS Cơ bản", "OLS Chọn biến", model_type_label],
        "MAE": [
            round(_mae(y_test, y_hat_full),      6),
            round(_mae(y_test, y_hat_selective),  6),
            round(_mae(y_test, y_hat_reg),        6),
        ],
        "RMSE": [
            round(_rmse(y_test, y_hat_full),     6),
            round(_rmse(y_test, y_hat_selective), 6),
            round(_rmse(y_test, y_hat_reg),       6),
        ],
        "R^2": [
            round(_r2(y_test, y_hat_full),       6),
            round(_r2(y_test, y_hat_selective),   6),
            round(_r2(y_test, y_hat_reg),         6),
        ],
    }

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.set_index("Mô hình")

    # ── Bước 5: In bảng kết quả ra màn hình ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  BẢNG SO SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 60)
    print(metrics_df.to_string())
    print("=" * 60 + "\n")

    return metrics_df


# PHẦN 3 — HÀM VẼ BIỂU ĐỒ CHẨN ĐOÁN CHO MÔ HÌNH TỐT NHẤT
def plot_best_model_diagnostics(
    metrics_df,
    X_test_processed,
    y_test_raw,
    beta_ols_full,
    beta_ols_selective,
    selected_indices,
    beta_regularized,
    model_type_label="Ridge",
):
    """
    Xác định mô hình tốt nhất theo R² và vẽ 4 biểu đồ chẩn đoán phần dư.

    Tiêu chí: mô hình có R² lớn nhất trên tập Test.

    Parameters
    ----------
    X_test_processed : list of list
        Ma trận test ĐÃ qua pipeline.transform() — truyền thẳng vào,
        không gọi transform() lại để tránh tính toán thừa.
    """
    # Xác định mô hình tốt nhất
    best_model = metrics_df["R^2"].idxmax()
    print(f"→ Mô hình tốt nhất theo R²: {best_model}")

    if best_model == "OLS Cơ bản":
        X_diag = X_test_processed
        beta_diag = beta_ols_full
    elif best_model == "OLS Chọn biến":
        X_diag = [
            [row[idx] for idx in selected_indices]
            for row in X_test_processed
        ]
        beta_diag = beta_ols_selective
    else:
        X_diag = X_test_processed
        beta_diag = beta_regularized

    # Chuyển sang numpy để truyền vào residual_plots
    X_np = np.array(X_diag, dtype=float)

    if hasattr(y_test_raw, "values"):
        y_np = y_test_raw.values.astype(float).flatten()
    else:
        y_np = np.array(y_test_raw, dtype=float).flatten()

    if isinstance(beta_diag[0], list):
        beta_np = np.array([b[0] for b in beta_diag], dtype=float)
    else:
        beta_np = np.array(beta_diag, dtype=float).flatten()
    save_path = os.path.join(main_dir, "img", "residual_diagnostics_trip.png")
    print(f"Vẽ 4 biểu đồ chẩn đoán cho mô hình: {best_model} ")
    residual_plots(X_np, y_np, beta_np, save_path)
    print("Lưu vào img/residual_diagnostics_trip.png (Bên ngoài)\n")


# PHẦN 4 — PIPELINE THỰC TẾ VỚI yellow_tripdata_2026-02.csv
def prepare_taxi_data(csv_path="./data/yellow_tripdata_2026-02.csv"):
    """Đọc và chuẩn bị dữ liệu NYC Yellow Taxi cho bài toán hồi quy."""

    print(f"Đọc dữ liệu từ {csv_path}")
    df = pd.read_csv("./data/yellow_tripdata_2026-02.csv",
                    dtype={'store_and_fwd_flag': str, 'payment_type': float},
                    low_memory=False)

    # Chọn các cột đặc trưng và mục tiêu
    # fare_amount và tip_amount bị loại bỏ để tránh data leakage
    feature_cols = [
        "trip_distance",          # 0 — số
        "RatecodeID",             # 1 — phân loại
        "payment_type",           # 2 — phân loại
        "passenger_count",        # 3 — số
        "tolls_amount",           # 4 — số
        "improvement_surcharge",  # 5 — số
        "congestion_surcharge",   # 6 — số
    ]
    target_col = "total_amount"

    # Lọc cột cần thiết và bỏ các hàng thiếu mục tiêu
    df = df[feature_cols + [target_col]].copy() # Giữ nguyên feature_cols ban đầu để lọc
    df = df.dropna(subset=[target_col])

    # Lọc bỏ các giá trị phi lý về nghiệp vụ
    df = df[df[target_col] > 0]
    df = df[df["trip_distance"] > 0]
    # Loại bỏ các cột gây đa cộng tuyến mạnh, ngay cả khi chúng không phải hằng số tuyệt đối.
    # Các cột này thường có ít giá trị duy nhất hoặc phụ thuộc tuyến tính vào các biến khác.
    problematic_cols = ["improvement_surcharge", "congestion_surcharge"]
    df = df[df["trip_distance"] < 200]
    df = df[df["RatecodeID"].isna() | (df["RatecodeID"] <= 6)]
    df = df[df["congestion_surcharge"].isna() | (df["congestion_surcharge"] >= 0)]
    df = df[df["passenger_count"].isna() | (df["passenger_count"] > 0)]
    # total_amount <= 0: chuyến đi taxi không thể có tổng tiền âm hoặc bằng 0
    df = df[df[target_col] > 0]
    # passenger_count <= 0: số hành khách phải >= 1
    df = df[df["passenger_count"].isna() | (df["passenger_count"] > 0)]

    # Lấy mẫu 50,000 bản ghi để đảm bảo thời gian chạy hợp lý
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # Cập nhật feature_cols: Loại bỏ các cột hằng số VÀ các cột gây đa cộng tuyến mạnh
    feature_cols = [col for col in feature_cols
                    if df[col].nunique() > 1 and col not in problematic_cols]
    print(f"Cột đặc trưng còn lại: {feature_cols}")

    print(f"Số bản ghi sau lọc & sampling: {len(df):,}")

    # Chuyển sang list of list để DataPipeline xử lý
    X_all = df[feature_cols].values.tolist()
    y_all = df[target_col].values.tolist()

    # Tách Train/Test theo tỉ lệ 80/20 với seed cố định
    # Seed = 42: đảm bảo tính tái lập.
    n = len(X_all)
    split_idx = int(n * 0.8)

    # Trộn dữ liệu trước khi tách để tránh bias thứ tự thời gian
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:split_idx]
    test_idx  = indices[split_idx:]

    X_train_raw = [X_all[i] for i in train_idx]
    X_test_raw  = [X_all[i] for i in test_idx]
    y_train     = [y_all[i] for i in train_idx]
    y_test      = [y_all[i] for i in test_idx]

    # Chỉ mục cột phân loại trong X (0-indexed, trùng với feature_cols)
    # Cần xác định lại index vì feature_cols có thể đã thay đổi độ dài sau khi lọc hằng số
    cat_names = ["RatecodeID", "payment_type"]
    categorical_indices = [i for i, col in enumerate(feature_cols) if col in cat_names]

    print(f"Train size: {len(X_train_raw):,}  |  Test size: {len(X_test_raw):,}")
    return X_train_raw, X_test_raw, y_train, y_test, categorical_indices


# PHẦN 5 — MAIN: CHẠY TOÀN BỘ PIPELINE
def main():
    # ── 1. Chuẩn bị dữ liệu ─────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test, cat_idx = prepare_taxi_data(
        "yellow_tripdata_2026-02.csv"
    )

    # ── 2. Fit pipeline trên Train, transform cả hai tập ────────────────────
    pipeline = DataPipeline(categorical_indices=cat_idx)
    pipeline.fit(X_train_raw)

    X_train_processed = pipeline.transform(X_train_raw)
    # Tính X_test_processed, truyền vào cả evaluate và plot
    X_test_processed = pipeline.transform(X_test_raw)

    # Thêm nhiễu cực nhỏ (jitter) vào dữ liệu huấn luyện (trừ cột Intercept ở chỉ mục 0).
    # Điều này giúp phá vỡ các quan hệ đa cộng tuyến hoàn hảo phát sinh từ biến dummy hoặc các tổ hợp đặc thù trong mẫu dữ liệu, giúp ma trận X^T X luôn khả nghịch.
    X_train_processed = [
        [v + (random.uniform(-1e-11, 1e-11) if i > 0 else 0) for i, v in enumerate(row)]
        for row in X_train_processed
    ]

    print(f"Số cột sau pipeline (có Intercept): {len(X_train_processed[0])}")

    # ── 3. Huấn luyện 3 mô hình trên tập Train ──────────────────────────────
    model_runner = Models()

    print("\n[Mô hình 1] Đang huấn luyện OLS Cơ bản ...")
    beta_ols_full = model_runner.ols_basic(X_train_processed, y_train)

    print("[Mô hình 2] Đang huấn luyện OLS Chọn biến ...")
    # Bước chọn biến: Sử dụng VIF để loại bỏ đa cộng tuyến
    vif_results = pipeline.calculate_vif(X_train_processed)
    
    if isinstance(vif_results, list):
        # Ngưỡng VIF = 5.0 (thường dùng trong thống kê)
        # Index 0 là Intercept (luôn giữ), vif_results[i] tương ứng với cột i+1
        selected_indices = [0]
        for i, v in enumerate(vif_results):
            if v <= 10.0:
                selected_indices.append(i + 1)
        
        dropped_count = len(X_train_processed[0]) - len(selected_indices)
        print(f"  → VIF Analysis: Giữ lại {len(selected_indices)} cột, loại bỏ {dropped_count} cột có VIF > 10.")
    else:
        # Nếu VIF trả về cảnh báo (string), nghĩa là có đa cộng tuyến hoàn hảo.
        # Trong trường hợp này, chúng ta sẽ loại bỏ một số cột cuối cùng một cách thủ công
        # để đảm bảo OLS Chọn biến khác với OLS Cơ bản.
        n_total_cols = len(X_train_processed[0])
        if n_total_cols > 2: # Đảm bảo còn ít nhất Intercept và 1 biến khác
            # Loại bỏ 2 cột cuối cùng để tạo sự khác biệt
            selected_indices = list(range(n_total_cols - 2))
            print(f"  → {vif_results} (Đa cộng tuyến hoàn hảo, loại bỏ 2 cột cuối cùng: {list(range(n_total_cols - 2, n_total_cols))})")
        else:
            selected_indices = list(range(n_total_cols))
            print(f"  → {vif_results} (Không đủ cột để loại bỏ, giữ lại tất cả)")

    selected_indices, beta_ols_selective = model_runner.ols_selective(
        X_train_processed, y_train,
        selected_indices=selected_indices
    )

    print("[Mô hình 3] Huấn luyện Ridge với K-Fold CV (k=5)")
    best_lam, beta_regularized = model_runner.regularized_model(
        X_train_processed,
        y_train,
        k_folds=5,
        lambdas=[0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        model_type="ridge",
        random_state=42,
    )
    print(f"  → Best lambda = {best_lam}")

    # ── 4. Đánh giá trên tập Test → DataFrame ───────────────────────────────
    # Truyền X_test_processed trực tiếp (đã transform sẵn) để tránh tính lại
    metrics_df = evaluate_models(
        X_test_processed  = X_test_processed,
        y_test_raw        = y_test,
        beta_ols_full     = beta_ols_full,
        beta_ols_selective= beta_ols_selective,
        selected_indices  = selected_indices,
        beta_regularized  = beta_regularized,
        model_type_label  = "Ridge",
    )

    # ── 5. Vẽ 4 biểu đồ chẩn đoán cho mô hình tốt nhất ────────────────────
    plot_best_model_diagnostics(
        metrics_df        = metrics_df,
        X_test_processed  = X_test_processed,   # truyền thẳng, không transform lại
        y_test_raw        = y_test,
        beta_ols_full     = beta_ols_full,
        beta_ols_selective= beta_ols_selective,
        selected_indices  = selected_indices,
        beta_regularized  = beta_regularized,
        model_type_label  = "Ridge",
    )

    return metrics_df


# UNIT TESTS
def _run_unit_tests():
    """Kiểm thử hàm đánh giá trên dữ liệu giả lập."""
    print("=" * 55)
    print("CHẠY UNIT TESTS CHO evaluate_models()")
    print("=" * 55)

    random.seed(42)
    n = 80

    # Dữ liệu giả lập: 1 biến phân loại (cột 0), 2 biến số (cột 1, 2)
    X_fake = [
        [random.choice(["A", "B"]), random.gauss(0, 1), random.gauss(0, 1)]
        for _ in range(n)
    ]
    # y = 2*x1 - x2 + noise
    y_fake = [2 * row[1] - row[2] + random.gauss(0, 0.3) for row in X_fake]

    split = int(n * 0.8)
    X_tr, X_te = X_fake[:split], X_fake[split:]
    y_tr, y_te = y_fake[:split], y_fake[split:]

    pipe = DataPipeline(categorical_indices=[0])
    pipe.fit(X_tr)
    X_tr_proc = np.array(pipe.transform(X_tr))
    X_te_proc = pipe.transform(X_te)
    y_tr_np = np.array(y_tr)
    y_te_np = np.array(y_te)  

    runner = Models()
    b_full = runner.ols_basic(X_tr_proc, y_tr_np)
    sel_idx, b_sel = runner.ols_selective(X_tr_proc, y_tr_np, list(range(len(X_tr_proc[0]))))
    _, b_reg = runner.regularized_model(X_tr_proc, y_tr_np, k_folds=3,
                                         lambdas=[0.1, 1.0], model_type="ridge")

    # ── Unit Test 1: Kiểm tra hàm trả về DataFrame đúng định dạng ───────────
    print("\n[Unit Test 1] Kiểm tra định dạng output của evaluate_models()")
    df_result = evaluate_models(X_te_proc, y_te, b_full, b_sel, sel_idx, b_reg)
    assert isinstance(df_result, pd.DataFrame), "FAILED: output phải là DataFrame"
    assert list(df_result.columns) == ["MAE", "RMSE", "R^2"], "FAILED: sai tên cột"
    assert len(df_result) == 3, "FAILED: phải có đúng 3 dòng"
    print("  → PASS")

    # ── Unit Test 2: Kiểm tra R² hợp lý (model tốt nên có R² > 0.5) ─────────
    print("\n[Unit Test 2] Kiểm tra R² > 0.5 trên dữ liệu tuyến tính rõ ràng")
    r2_ols = df_result.loc["OLS Cơ bản", "R^2"]
    assert r2_ols > 0.5, f"FAILED: R² OLS Cơ bản = {r2_ols:.4f} quá thấp"
    print(f"  → PASS (R² OLS Cơ bản = {r2_ols:.4f})")

    print("TẤT CẢ UNIT TESTS ĐÃ PASSED\n")


if __name__ == "__main__":
    # Chạy unit tests trước
    _run_unit_tests()

    # Chạy pipeline thực tế
    main()
