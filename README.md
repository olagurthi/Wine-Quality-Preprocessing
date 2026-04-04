# 🍷 Wine Quality — Preprocessing 

## Project Overview

This project performs a full preprocessing and feature engineering pipeline on the **WineQT dataset**, which contains physicochemical measurements of red wine samples and their quality ratings (scored 3–8).

The goal is to clean and prepare the data so it is ready for machine learning models.

## Steps Performed in `main.py`

### 1. Load the Dataset
- Loaded `WineQT.csv` using `pandas`
- Printed shape and basic statistics

### 2. Handle Missing Data
- Checked all columns with `isnull().sum()`
- Visualised missingness with a heatmap
- Strategy: impute with **median** if any missing values found (robust to outliers)
- Result: no missing values found in this dataset

### 3. Handle Duplicates
- Detected duplicates with `duplicated().sum()`
- Removed duplicate rows with `drop_duplicates()`
- Dropped the `Id` column (row identifier, not a feature)

### 4. Detect and Treat Outliers
- Method: **IQR (Interquartile Range)**
- Formula: lower = Q1 − 1.5×IQR, upper = Q3 + 1.5×IQR
- Treatment: **Capping / Winsorization** — extreme values are clipped to the bounds instead of removed, preserving all rows
- Produced boxplots before and after treatment

### 5. Encode Categorical Variables
- All features are numeric — no one-hot encoding needed
- The `quality` target was **binarised**:
  - `quality >= 6` → `1` (Good)
  - `quality < 6` → `0` (Low)
- New column: `quality_label`

### 6. Split Data
- Split into **80% train / 20% test** using `train_test_split`
- Used `stratify=y` to keep the same class ratio in both sets
- Split is done **before scaling** to prevent data leakage

### 7. Feature Scaling
- Method: **StandardScaler** (Z-score normalisation)
- Scaler is **fit on training data only**, then applied to both train and test
- This prevents information from the test set leaking into training

---

## How to Run

1. Make sure `WineQT.csv` and `main.py` are in the same folder
2. Install dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Run the script:
```
python main.py
```

---

## Output

- Printed summaries for each preprocessing step
- Visualisations: missing value heatmap, boxplots before/after outlier treatment, class distribution bar chart
- Final scaled datasets: `X_train_scaled`, `X_test_scaled`, `y_train`, `y_test`

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Scaling and train/test split |

---

## Author
Lab 3 — Preprocessing & Feature Engineering
