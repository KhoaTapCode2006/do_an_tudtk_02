import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

print("KHAO SAT DU LIEU (EDA)")
print("Reading data...")
df = pd.read_parquet("./data/yellow_tripdata_2026-02.parquet")
print(f"Data shape: {df.shape}")

print("\n1. Thong ke mo ta (Descriptive Statistics):")
desc = df.describe().T
desc['median'] = df.median(numeric_only=True)
print(desc[['mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n2. Kiem tra du lieu trung lap:")
duplicates = df.duplicated().sum()
print(f"So dong trung lap: {duplicates}")
print(f"Ti le trung lap: {duplicates / len(df) * 100:.4f}%")

print("\n3. Phan tich missing values:")
missing = df.isnull().sum()
missing_ratio = missing / len(df) * 100
missing_df = pd.DataFrame({'Missing_Values': missing, 'Ratio(%)': missing_ratio})
print(missing_df[missing_df['Missing_Values'] > 0])

print("\n4. Phat hien outliers (Su dung phuong phap IQR):".encode('ascii', 'ignore').decode())
numeric_cols = df.select_dtypes(include=['number']).columns

# Chọn một số biến quan trọng như trip_distance, fare_amount, total_amount để phát hiện outlier
key_cols = ['trip_distance', 'fare_amount', 'total_amount']
for col in key_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Bien {col}: So luong outliers = {len(outliers)} ({len(outliers)/len(df)*100:.2f}%), Nguong = [{lower_bound:.2f}, {upper_bound:.2f}]")

print("\nDang tao bieu do...")
os.makedirs("eda_plots", exist_ok=True)

# Lấy mẫu 10% dữ liệu để vẽ cho nhanh nếu dữ liệu quá lớn
df_sample = df.sample(frac=0.1, random_state=42) if len(df) > 100000 else df

for col in key_cols:
    if col in df_sample.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_sample[col], bins=50, kde=True)
        plt.title(f'Histogram of {col}')
        plt.savefig(f'eda_plots/hist_{col}.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df_sample[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(f'eda_plots/box_{col}.png')
        plt.close()

# Heatmap
plt.figure(figsize=(12, 10))
corr = df_sample.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('eda_plots/heatmap.png')
plt.close()

print("Hoan tat EDA! Cac bieu do da duoc luu trong thu muc 'eda_plots'.")
