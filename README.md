# 🏪 Rossmann Store Sales Forecasting

This repository contains our solution for the **Rossmann Store Sales Forecasting** challenge, where the goal is to predict daily sales for more than **1,000 stores across Europe**.  
The project leverages **gradient boosting ensembles**, advanced **feature engineering**, and **automated hyperparameter optimization (Optuna)** to achieve state-of-the-art performance.

---

## 📊 Dataset
The dataset is based on the [Rossmann Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales).  
It includes:
- Store identifiers
- Daily sales
- Promotions
- Holidays and calendar information
- Store metadata (assortment, competition distance, etc.)

---

## 🔬 Methodology

### 1. Time Series Analysis
- ADF and KPSS tests confirmed that global sales are **non-stationary** due to trend and seasonality.
- Per-store analysis showed mixed levels of stationarity → univariate ARIMA models were deemed impractical.
- Decision: **use global tree-based models** with feature engineering.

### 2. Feature Engineering
- **Calendar features**: year, month, week, weekday, holidays.  
- **Lag features**: sales shifted by `[7, 14, 28]` days.  
- **Rolling statistics**: moving averages, standard deviations, medians, min/max windows.  
- **Target transformation**: applied `np.log1p(Sales)` to stabilize variance.

### 3. Models
We trained three gradient boosting models:
- **LightGBM**
- **CatBoost**
- **XGBoost**

Each was tuned with **Optuna**, then combined via a simple **blending strategy**:

$$
y_{blend} = \frac{y_{lgb} + y_{cat} + y_{xgb}}{3}
$$

---

## ⚙️ Best Hyperparameters

- **LightGBM**:  
  `learning_rate=0.141, num_leaves=55, max_depth=6`

- **CatBoost**:  
  `learning_rate=0.168, depth=10, l2_leaf_reg=7.69`

- **XGBoost**:  
  `learning_rate=0.077, max_depth=8, subsample=0.95, colsample_bytree=0.85`

---

## 📈 Results

| Model         | RMSPE (validation) |
|---------------|---------------------|
| LightGBM      | ~8.5–8.6%           |
| CatBoost      | ~8.4–8.5%           |
| XGBoost       | ~8.5–8.6%           |
| **Blended**   | **8.14%**           |

- **Baseline (before tuning):** 9.03% RMSPE  
- **Final blended (Optuna):** **8.14% RMSPE**

➡️ This improvement of nearly **1 percentage point** is highly significant in retail forecasting.

---

## 🚀 Reproducibility

**1. Local environment**  
-Clone this repository:
   ```bash
   git clone https://github.com/MVasquez95/Rossman-Store-Sales.git
   cd rossmann-forecasting
   ```

-Install dependencies:
```bash
    pip install -r requirements.txt
```

-Run the Jupyter notebook:
```bash
    jupyter notebook notebook.ipynb
```
**2. Kaggle Notebook** 
You can run the full pipeline directly on Kaggle without local setup:
[View on Kaggle](https://www.kaggle.com/code/crowwick/end-to-end-rossmann-feature-engineering-optuna).
⚠️ **Note:** Due to slight differences in package versions between local (`requirements.txt`) and Kaggle environments, the validation score may vary marginally (e.g., **8.1358% local vs. 8.1671% Kaggle**).  
This variation is expected and does not affect the overall conclusions.

## 📝 Conclusion

A global gradient boosting ensemble with feature engineering clearly outperformed classical time series methods.
Automated hyperparameter optimization (Optuna) was essential to reach top-tier performance.
The blended model is competitive for **large-scale, real-world retail forecasting**.

## 🙌 Acknowledgments

[Kaggle Rossmann competition](https://www.kaggle.com/c/rossmann-store-sales).
Libraries: **LightGBM, XGBoost, CatBoost, Optuna, Pandas, NumPy, Scikit-learn**