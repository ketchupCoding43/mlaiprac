import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Binarizer

# ==============================
# 1. CREATE SAMPLE DATASET
# ==============================
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Gender": ["Female", "Male", "Female", "Male", "Female"],
    "City": ["Mumbai", "Delhi", "Mumbai", "Chennai", "Delhi"],
    "Age": [25, 30, 35, 28, 22],
    "Salary": [50000, 60000, 55000, 52000, 48000]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)


# ==============================
# 2. LABEL ENCODING (Categorical → Numbers)
# ==============================

le = LabelEncoder()

df["Gender_Encoded"] = le.fit_transform(df["Gender"])
df["City_Encoded"] = le.fit_transform(df["City"])

print("\nAfter Label Encoding:\n", df)


# ==============================
# 3. FEATURE SCALING
# ==============================

# Standardization (mean=0, std=1)
scaler_std = StandardScaler()
df[["Age_Std", "Salary_Std"]] = scaler_std.fit_transform(df[["Age", "Salary"]])

# Normalization (0–1 range)
scaler_mm = MinMaxScaler()
df[["Age_Norm", "Salary_Norm"]] = scaler_mm.fit_transform(df[["Age", "Salary"]])

print("\nAfter Scaling:\n", df)


# ==============================
# 4. BINARIZATION
# ==============================

# Convert Salary into 0/1 based on threshold
binarizer = Binarizer(threshold=52000)
df["Salary_Binary"] = binarizer.fit_transform(df[["Salary"]])

print("\nAfter Binarization:\n", df)


# ==============================
# 5. FINAL DATASET READY FOR ML
# ==============================
print("\nFinal Processed Dataset:\n", df)
