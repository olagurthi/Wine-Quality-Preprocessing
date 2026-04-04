import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Libraries loaded successfully.")

# ─────────────────────────────────────────
# 1. LOAD THE DATASET
# ─────────────────────────────────────────
df = pd.read_csv("WineQT.csv")

print("\n── 1. LOAD ──────────────────────────────")
print("Shape:", df.shape)
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())

# ─────────────────────────────────────────
# 2. CHECK AND HANDLE MISSING DATA
# ─────────────────────────────────────────
print("\n── 2. MISSING DATA ──────────────────────")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Missing %": missing_pct.round(2)
})
print(missing_df)
print(f"Total missing cells: {df.isnull().sum().sum()}")

# Visualise missingness
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.tight_layout()
plt.show()

# Impute with median if any missing values exist
cols_with_missing = missing[missing > 0].index.tolist()
if cols_with_missing:
    for col in cols_with_missing:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Imputed '{col}' with median = {median_val}")
else:
    print("No missing values found. No imputation needed.")

print(f"Missing values after handling: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. CHECK AND HANDLE DUPLICATES
# ─────────────────────────────────────────
print("\n── 3. DUPLICATES ────────────────────────")
num_duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {num_duplicates}")
print(f"Shape before: {df.shape}")

if num_duplicates > 0:
    print("\nSample duplicate rows:")
    print(df[df.duplicated(keep=False)].head(6))

df = df.drop_duplicates(keep="first").reset_index(drop=True)

# Drop the Id column — it is just a row identifier, not a feature
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    print("Dropped the 'Id' column (not a feature).")

print(f"Shape after removing duplicates: {df.shape}")
print(f"Rows removed: {num_duplicates}")

# ─────────────────────────────────────────
# 4. DETECT AND TREAT OUTLIERS (IQR)
# ─────────────────────────────────────────
print("\n── 4. OUTLIERS ──────────────────────────")
feature_cols = [c for c in df.columns if c != "quality"]

# Boxplots BEFORE treatment
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xticks([])
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Boxplots Before Outlier Treatment", fontsize=13)
plt.tight_layout()
plt.show()

# IQR capping (Winsorization)
outlier_report = []
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report.append({
        "Feature": col,
        "Lower Bound": round(lower, 4),
        "Upper Bound": round(upper, 4),
        "Outliers Found": n_outliers
    })
    df[col] = df[col].clip(lower=lower, upper=upper)

report_df = pd.DataFrame(outlier_report)
print("Outlier detection & capping summary (IQR method):")
print(report_df.to_string(index=False))

# Boxplots AFTER treatment
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xticks([])
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Boxplots After Outlier Treatment (Capping)", fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 5. ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────
print("\n── 5. ENCODING ──────────────────────────")
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Categorical columns: {cat_cols if cat_cols else 'None — all columns are numeric'}")

print("\nTarget variable (quality) distribution:")
print(df["quality"].value_counts().sort_index())

# Binarise quality: >= 6 → 1 (good), < 6 → 0 (low)
df["quality_label"] = (df["quality"] >= 6).astype(int)
print("\nBinarised quality_label distribution:")
print(df["quality_label"].value_counts())
print("  0 = Low quality (< 6)   |   1 = Good quality (>= 6)")

plt.figure(figsize=(5, 4))
df["quality_label"].value_counts().plot(kind="bar", color=["#e74c3c", "#2ecc71"], edgecolor="black")
plt.title("Quality Label Distribution")
plt.xlabel("Label (0 = Low, 1 = Good)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 6. SPLIT DATA (before scaling!)
# ─────────────────────────────────────────
print("\n── 6. SPLIT DATA ────────────────────────")
X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

print("Features shape:", X.shape)
print("Target shape  :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain/Test split (80/20):")
print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")
print(f"\nClass balance in y_train:")
print(y_train.value_counts(normalize=True).round(3))

# ─────────────────────────────────────────
# 7. FEATURE SCALING (StandardScaler)
# ─────────────────────────────────────────
print("\n── 7. FEATURE SCALING ───────────────────")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train only
X_test_scaled = scaler.transform(X_test)  # transform only on test

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Scaling complete (StandardScaler — fit on train only).")
print("\nX_train_scaled sample (first 3 rows):")
print(X_train_scaled.head(3))

# ─────────────────────────────────────────
# FINAL CHECK
# ─────────────────────────────────────────
print("\n── FINAL CHECK ──────────────────────────")
print(f"X_train_scaled missing values : {X_train_scaled.isnull().sum().sum()}")
print(f"X_test_scaled  missing values : {X_test_scaled.isnull().sum().sum()}")
print(f"X_train_scaled shape          : {X_train_scaled.shape}")
print(f"X_test_scaled  shape          : {X_test_scaled.shape}")
print("\nPreprocessing pipeline complete. Data is model-ready.")
