# Multivariate Linear Regression: Regularization, Scaling & Feature Selection

## Overview

This project explores **Multivariate Linear Regression** on the Global Air Pollution dataset, with a particular focus on **feature selection and the impact of feature scaling on regularized regression models**.

The project begins with a standard Multivariate Linear Regression model and then compares three regularization approaches:

- **LASSO (L1 Regularization)**
- **Ridge (L2 Regularization)**
- **Elastic Net (L1 + L2 Regularization)**

Each regularized model is evaluated using both **StandardScaler** and **RobustScaler** to study whether the choice of scaling method changes model performance or feature selection.

The main objective is not simply to obtain the highest prediction score, but to understand how **regularization and feature scaling influence the coefficients and selected features**.

---

## Dataset

The experiments use the **Global Air Pollution Dataset**.

### Features

The following pollutant-related AQI variables were used as predictors:

- `CO AQI Value`
- `Ozone AQI Value`
- `NO2 AQI Value`
- `PM2.5 AQI Value`
###Link of the dataset
https://www.kaggle.com/datasets/sazidthe1/global-air-pollution-data
### Target

- `AQI Value`

The project uses multiple independent variables simultaneously, making this a **multivariate linear regression** problem.

---

## Project Objective

The main aim of this project is:

> **To study feature selection using regularized regression and investigate how different feature-scaling methods affect the resulting coefficients, selected features, and model performance.**

The project compares:

```text
Multivariate Linear Regression
          ↓
      LASSO
          ↓
       Ridge
          ↓
    Elastic Net
          ↓
For each regularized model:
StandardScaler vs RobustScaler
```

---

# 1. Multivariate Linear Regression

A baseline Multivariate Linear Regression model was first developed using all four pollutant-related predictors.

The model achieved approximately:

- **MAE:** 4.8038
- **MSE:** 76.9921
- **RMSE:** 8.7745
- **R²:** 0.97584

This baseline provides a reference point for evaluating the effect of regularization.

---

# 2. Feature Scaling

Feature scaling was applied before LASSO, Ridge, and Elastic Net because these models use coefficient-based regularization.

Two scaling methods were investigated.

## StandardScaler

Standardization uses the mean and standard deviation:

\[
Z = rac{X-\mu}{\sigma}
\]

It places features on a comparable scale based on their mean and standard deviation.

## RobustScaler

RobustScaler uses the **median** and **interquartile range (IQR)**:

\[
X_{scaled} = rac{X-	ext{Median}}{IQR}
\]

where:

\[
IQR = Q3-Q1
\]

### Why was RobustScaler considered?

The dataset contains extreme observations, so RobustScaler was investigated because it uses the **median and IQR**, which are less influenced by extreme values than the mean and standard deviation.

The goal was therefore not to assume that RobustScaler is always better, but to test whether an outlier-resistant scaling method changes the behaviour of the regularized models.

---

# 3. LASSO Regression

LASSO stands for **Least Absolute Shrinkage and Selection Operator**.

It uses **L1 regularization**:

\[
	ext{Loss} + \lambda\sum|eta_j|
\]

The L1 penalty can shrink some coefficients exactly to zero.

Therefore:

> **LASSO can perform feature selection.**

### LASSO Results

With the chosen regularization strength (`alpha = 0.1`), the LASSO experiments selected:

```text
Ozone AQI Value
NO2 AQI Value
PM2.5 AQI Value
```

while the coefficient of:

```text
CO AQI Value → 0
```

was reduced to zero.

This was observed with the StandardScaler experiment as well as the RobustScaler experiment.
### Observation
| Scaling | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| Standard | **4.7831** | 77.0575 | 8.7782 | 0.975819 |
| Robust | 4.7872 | **77.0230** | **8.7763** | **0.975829** |

Lasso produced almost identical results with both scaling techniques. Robust Scaling performed slightly better for MSE, RMSE, and R², while Standard Scaling had a slightly lower MAE.
### Key Learning

LASSO can remove features from the model by shrinking their coefficients to exactly zero. This makes it particularly useful when the goal is **feature selection**, not just prediction.

---

# 4. Ridge Regression

Ridge Regression uses **L2 regularization**:

\[
	ext{Loss} + \lambda\sumeta_j^2
\]

Unlike LASSO, Ridge generally shrinks coefficients toward zero without forcing them to exactly zero.

### Observation

The Ridge experiments produced performance extremely close to the baseline Multivariate Linear Regression model.

| Scaling | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| Standard | **4.8037** | **76.9921** | **8.7745** | **0.975839** |
| Robust | 4.8037 | 76.9921 | 8.7745 | 0.975839 |

Ridge produced virtually identical results with Standard and Robust Scaling. Standard Scaling was marginally better, but the difference was negligible.

### Key Learning

Ridge is useful for **coefficient shrinkage and regularization**, but it does not provide the same direct feature-selection behaviour as LASSO.

---

# 5. Elastic Net

Elastic Net combines **L1 and L2 regularization**.

It can be viewed conceptually as:

\[
	ext{L1 regularization} + 	ext{L2 regularization}
\]

The experiments used:

```python
ElasticNet(alpha=0.1, l1_ratio=0.5)
```

Here:

- `alpha` controls the overall regularization strength.
- `l1_ratio` controls the balance between L1 and L2 regularization.

### Observations

| Scaling | MAE ↓ | MSE ↓ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|---:|
| Standard | 5.0986 | 85.4715 | 9.2451 | 0.973178 |
| Robust | **4.9135** | **80.7483** | **8.9860** | **0.974660** |

Elastic Net showed a clearer improvement with Robust Scaling. It achieved lower MAE, MSE, and RMSE and a higher R² score than its Standard Scaling counterpart.

For the chosen parameters, the RobustScaler version performed better than the StandardScaler version, although both were below the strongest Linear Regression/Ridge results.

---

# 6. Impact of Scaling on Feature Selection

One of the main questions explored in this project was:

> **Does changing the feature-scaling method affect which features are selected by a regularized model?**

For the LASSO experiments, both scaling methods produced the same selected features:

```text
                  StandardScaler     RobustScaler

CO AQI Value             0                  0
Ozone AQI Value       Selected            Selected
NO2 AQI Value         Selected            Selected
PM2.5 AQI Value       Selected            Selected
```

Therefore, for this particular dataset and the chosen `alpha`:

> **Changing from StandardScaler to RobustScaler did not change the set of features selected by LASSO.**

However, the actual coefficient values changed because the two scalers transform the features differently.

For example, the LASSO coefficient values were different under StandardScaler and RobustScaler even though the same features were retained.

### Important Insight

Scaling can influence the numerical values of regularized coefficients, and in other datasets or with different regularization strengths it can potentially affect which coefficients reach zero.

Thus, scaling should be considered as part of the modelling pipeline rather than treated as an independent step.

---

# 7. Model Comparison

| Model | Scaling | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Multivariate Linear Regression | None | 4.8038 | 8.7745 | 0.97584 |
| LASSO | StandardScaler | 4.7831 | 8.7782 | 0.97582 |
| LASSO | RobustScaler | 4.7872 | 8.7763 | 0.97583 |
| Ridge | StandardScaler | ~4.8037 | ~8.7745 | ~0.97584 |
| Ridge | RobustScaler | 4.8037 | 8.7745 | 0.97584 |
| Elastic Net | StandardScaler | 5.0986 | 9.2451 | 0.97318 |
| Elastic Net | RobustScaler | 4.9135 | 8.9860 | 0.97466 |

> The very small numerical differences between the Linear Regression and Ridge models indicate that their predictive performance is practically similar for this dataset.

---

## 🔎 Key Findings

### Scaling

- Standard and Robust Scaling produced very similar results for Lasso and Ridge.
- Robust Scaling produced a more noticeable improvement for Elastic Net.
- Robust Scaling is generally preferred when outliers are present because it uses the median and IQR and is less sensitive to extreme values.
- Based on the overall comparison, Robust Scaling showed a slightly better overall performance across the evaluated regularized models.

### Regression Models

- Ridge + Standard Scaling achieved the lowest MSE and RMSE and the highest R² among the tested combinations.
- Lasso + Standard Scaling achieved the lowest MAE.
- Elastic Net performed better with Robust Scaling than with Standard Scaling.
- The differences between Ridge and Lasso were very small.

---

---

---
## Conclusion

This study demonstrates how multivariate linear regression can be extended through regularization and feature scaling to gain deeper insights into model behaviour. While the baseline regression provided strong predictive performance, introducing LASSO, Ridge, and Elastic Net highlighted the distinct roles of each method. LASSO effectively performed feature selection by eliminating the `CO AQI Value` predictor, Ridge primarily shrank coefficients without exclusion, and Elastic Net offered a balance of both approaches, though its performance was sensitive to hyperparameter choices.

The comparison between StandardScaler and RobustScaler revealed that, for this dataset, the choice of scaling method produced only minor differences in predictive accuracy. Importantly, both scalers led LASSO to select the same set of features, though the magnitude of coefficients varied. This underscores that scaling should be considered an integral part of the modelling pipeline, as its influence may become more pronounced in datasets with stronger skewness or extreme outliers.

Overall, the findings emphasize that the value of regularized regression lies not only in improving prediction accuracy but also in enhancing interpretability. By examining how regularization and scaling affect coefficients and feature selection, practitioners can better understand the structure of their data and make more informed modelling decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Structure

```text
.
├── README.md
├── Multivariate Regression.ipynb
├── Lasso using standard scale.ipynb
├── Lasso using roboust scale.ipynb
├── Ridge using standard scalar.ipynb
├── Ridge using roboust scaler.ipynb
├── Elasric net using standard scalar.ipynb
└── Elastic net using roboust.ipynb
```
##Author
Vidya Nayak
