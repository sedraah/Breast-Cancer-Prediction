# Breast Cancer Classification: Logistic Regression with EDA

## Overview

A binary classification project using the scikit-learn Breast Cancer Wisconsin dataset. The workflow covers exploratory data analysis with interactive Plotly visualisations, feature selection based on target correlation, and a Logistic Regression classifier evaluated with a confusion matrix and full classification report.

**Data source:** [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) — loaded directly via `sklearn.datasets.load_breast_cancer`


## Dataset

| Property | Detail |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer(as_frame=True)` |
| Rows | 569 |
| Columns | 31 (30 features + 1 target) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target column | `target` (0 = malignant, 1 = benign) |

### Class distribution

| Class | Count | Proportion |
|---|---|---|
| 1 — Benign | 357 | 62.7% |
| 0 — Malignant | 212 | 37.3% |

The dataset has a moderate class imbalance (~1.7:1 benign-to-malignant ratio).


## Workflow

### Exploratory Data Analysis

**Univariate & bivariate analysis:**
- Histogram of `log10(mean radius)` overlaid by class, reveals distributional separation between malignant and benign tumours.
- Scatter plot of `mean radius` vs `mean texture`, coloured by class.
- Box plot of `worst area` grouped by class, malignant tumours show substantially higher and more variable worst-area values.
- Bar chart of mean `worst area` per class.

**Correlation analysis:**
- Heatmap of the 10 mean-feature correlations.
- Full correlation of all features with the target, sorted descending.

**Top negatively correlated features with target** (higher magnitude = stronger association with malignancy)



### Feature Selection & Modelling

**Selected features** (chosen based on target correlation):

```
mean texture, mean compactness, mean concavity, mean area,
mean radius, mean perimeter, mean concave points
```


### Model performance

**Accuracy: 0.8947**

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — Malignant | 0.91 | 0.83 | 0.87 | 47 |
| 1 — Benign | 0.89 | 0.94 | 0.91 | 67 |
| **Accuracy** | | | **0.89** | 114 |
| Macro avg | 0.90 | 0.89 | 0.89 | 114 |
| Weighted avg | 0.90 | 0.89 | 0.89 | 114 |

The model performs slightly better on the benign class (higher recall of 0.94) than on malignant cases (recall 0.83). In a clinical context, the lower malignant recall means some cancerous cases are missed — a potential area for improvement through threshold tuning or resampling.


## Key Findings

- **Radius, perimeter, area, and concave points** are the strongest predictors of malignancy, all show large negative correlations with the target label.
- **Worst-area distributions** clearly separate the two classes in box plots, with malignant tumours producing larger and more variable measurements.
- **Logistic Regression on 7 selected features** achieves 89.5% accuracy without any scaling, demonstrating that the chosen features carry strong linear separability.
- **Benign recall (0.94) > Malignant recall (0.83):** the model is more conservative about flagging malignancy, which in practice means some cancer cases are classified as benign. Applying `StandardScaler` and threshold tuning could improve sensitivity for the malignant class.

