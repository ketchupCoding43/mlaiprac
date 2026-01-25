import pandas as pd
import numpy as np

# ==============================
# 1. LOAD CSV DATASET
# ==============================
file_path = "messy_dataset.csv"
df = pd.read_csv(file_path)

print("Original Data:")
print(df.head())

# ==============================
# 2. FIX COLUMN NAME FORMATTING (DO THIS FIRST!)
# ==============================
df.columns = df.columns.str.strip()   # remove extra spaces

# ==============================
# 3. HANDLE MISSING VALUES
# ==============================
print("\nMissing Values:")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill numerical with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==============================
# 4. FIX INCONSISTENT FORMATTING
# ==============================
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Standardize gender
if "gender" in df.columns:
    df["gender"] = df["gender"].replace({
        "male": "m",
        "female": "f",
        "m ": "m",
        "f ": "f"
    })

# ==============================
# 5. HANDLE OUTLIERS (IQR)
# ==============================
def cap_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data[column] = np.where(data[column] < lower, lower, data[column])
    data[column] = np.where(data[column] > upper, upper, data[column])

for col in num_cols:
    cap_outliers_iqr(df, col)

# ==============================
# 6. FINAL OUTPUT
# ==============================
print("\nCleaned Data:")
print(df.head())

df.to_csv("cleaned_data.csv", index=False)
print("\n Cleaned dataset saved!")
