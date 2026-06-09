
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np # Dùng để ép kiểu array và flatten
from part1.ols_implementation import ols_fit
from part1.ridge_lasso import ridge_fit

class Models:
    """
    Class đóng gói 3 mô hình hồi quy: OLS Cơ bản, OLS Chọn biến, Ridge/Lasso.
    """
    def __init__(self):
        pass

    def _matrix_vector_multiply(self, X, beta):
        """Phép nhân ma trận X (n x p) với vector cột beta (p x 1).

        Trả về vector danh sách 1 chiều (n phần tử).
        """
        # Đảm bảo beta là danh sách 1 chiều 
        b = [item[0] if isinstance(item, list) else item for item in beta]
        y_pred = []
        for row in X:
            pred = sum(x_ij * b_j for x_ij, b_j in zip(row, b))
            y_pred.append(pred)
        return y_pred
    
    def _mean(self, values):
        """Tính trung bình cộng của một danh sách số."""
        return sum(values) / len(values) if values else 0.0
    
    # ==========================================
    # MÔ HÌNH 1: OLS CƠ BẢN
    # ==========================================
    def ols_basic(self, X_train_processed, y_train):
        """
        Huấn luyện OLS trên toàn bộ biến.
        """
        X_subset_np = np.array(X_train_processed)
        
        beta_hat, sigma2 = ols_fit(X_subset_np, y_train)
        # Đảm bảo output luôn là vector cột (p+1, 1)
        beta_flat = np.asarray(beta_hat).flatten().tolist()
        return [[float(b)] for b in beta_flat]

    # ==========================================
    # MÔ HÌNH 2: OLS CHỌN BIẾN (FEATURE SELECTION)
    # ==========================================
    def ols_selective(self, X_train_processed, y_train, selected_indices):
        """
        Huấn luyện OLS trên một tập hợp các biến được chỉ định (Dựa trên VIF/P-value).
        """
        # Giữ cột 0 (Intercept)
        if 0 not in selected_indices:
            selected_indices = [0] + list(selected_indices)
        
        # Lọc ma trận X chỉ lấy các cột trong selected_indices
        X_subset = []
        for row in X_train_processed:
            subset_row = [row[idx] for idx in selected_indices]
            X_subset.append(subset_row)

        X_subset_np = np.array(X_subset)
        
        beta_hat, sigma2 = ols_fit(X_subset_np, y_train)

        beta_flat = np.asarray(beta_hat).flatten().tolist()
        return selected_indices, [[float(b)] for b in beta_flat]

    # ==========================================
    # MÔ HÌNH 3: RIDGE / LASSO VỚI K-FOLD CV
    # ==========================================
    def regularized_model(
        self,
        X_train_processed,
        y_train,
        k_folds=5,
        lambdas=None,
        model_type="ridge",
        random_state=42,
    ):
        """Huấn luyện Ridge hoặc Lasso, tự động chọn lambda bằng K-Fold CV viết

        bằng Python thuần.
        """
        if lambdas is None:
            lambdas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

        # Đảm bảo y_train là mảng 1 chiều thuần túy để lặp
        if isinstance(y_train[0], list):
            y_flat = [row[0] for row in y_train]
        else:
            y_flat = [val for val in y_train]

        n = len(y_flat)

        # Chi dữ liệu thành K-Folds
        indices = list(range(n))
        random.seed(random_state)
        random.shuffle(indices)  # Trộn ngẫu nhiên danh sách index

        # Chia index thành k mảng con (folds) 
        folds = []
        fold_size = n // k_folds
        remainder = n % k_folds
        current = 0
        for i in range(k_folds):
            size = fold_size + (1 if i < remainder else 0)
            folds.append(indices[current : current + size])
            current += size

        best_lambda = None
        min_mse = float("inf")

        # Duyệt qua từng lambda ứng viên
        for lam in lambdas:
            mse_list = []

            # Vòng lặp K-Fold
            for i in range(k_folds):
                val_indices = set(folds[i])

                X_tr, y_tr = [], []
                X_val, y_val = [], []

                for idx in range(n):
                    # Lấy dòng dữ liệu hiện tại
                    row_data = X_train_processed[idx]
                    
                    # Ép về list 
                    if hasattr(row_data, "tolist"):
                        row_data = row_data.tolist()
                    else:
                        row_data = list(row_data)

                    # Phân bổ vào tập Validation hoặc Train tương ứng
                    if idx in val_indices:
                        X_val.append(row_data)
                        y_val.append(y_flat[idx])
                    else:
                        X_tr.append(row_data)
                        y_tr.append(y_flat[idx])

                # Huấn luyện mô hình tương ứng trên fold hiện tại
                if model_type.lower() == "ridge":
                    beta_fold = ridge_fit(X_tr, y_tr, lam)

                # Dự đoán trên tập Validation 
                y_pred = self._matrix_vector_multiply(X_val, beta_fold)

                # Tính sai số MSE trên tập Validation
                mse = sum((y_v - y_p) ** 2 for y_v, y_p in zip(y_val, y_pred)) / len(y_val)
                mse_list.append(mse)

            # Tính trung bình MSE của tất cả các fold
            avg_mse = self._mean(mse_list)

            # Lưu lại lambda tốt nhất có độ lỗi thấp nhất
            if avg_mse < min_mse:
                min_mse = avg_mse
                best_lambda = lam

        # Fit lại trên toàn bộ data train với lambda tìm dược 
        if model_type.lower() == "ridge":
            beta_final = ridge_fit(X_train_processed, y_flat, best_lambda)

        beta_flat = np.asarray(beta_final).flatten().tolist()
        return best_lambda, [[float(b)] for b in beta_flat]