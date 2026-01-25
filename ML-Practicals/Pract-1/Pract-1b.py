import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("messy_dataset.csv")
df.columns = df.columns.str.strip()

# Create folder to store graphs
import os
os.makedirs("graphs", exist_ok=True)

print("First 5 rows:\n", df.head())

# ==============================
# 2. DESCRIPTIVE SUMMARY STATS
# ==============================
print("\n--- Statistical Summary (Numerical) ---")
print(df.describe())

print("\n--- Statistical Summary (Categorical) ---")
print(df.describe(include='object'))

print("\n--- Missing Values ---")
print(df.isnull().sum())


# ==============================
# 3. UNIVARIATE ANALYSIS
# ==============================

# Histogram
plt.figure()
df['Age'].hist(bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.savefig("graphs/age_distribution.png")
plt.show()

# Boxplot
plt.figure()
sns.boxplot(x=df['Salary'])
plt.title("Salary Boxplot")
plt.savefig("graphs/salary_boxplot.png")
plt.show()

# Countplot
plt.figure()
sns.countplot(x='Gender', data=df)
plt.title("Gender Count")
plt.savefig("graphs/gender_count.png")
plt.show()


# ==============================
# 4. BIVARIATE ANALYSIS
# ==============================

# Scatter Plot
plt.figure()
plt.scatter(df['Age'], df['Salary'])
plt.title("Age vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.savefig("graphs/age_vs_salary.png")
plt.show()

# Bar Plot
plt.figure()
sns.barplot(x='Gender', y='Salary', data=df)
plt.title("Average Salary by Gender")
plt.savefig("graphs/avg_salary_by_gender.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("graphs/correlation_heatmap.png")
plt.show()


# ==============================
# 5. FEATURES & TARGET
# ==============================
print("\n--- Feature & Target Suggestion ---")
print("Features (X): Age, Gender, City")
print("Target (y): Salary")
