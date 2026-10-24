# Customer_Churn_Prediction
End-to-end Telco Churn Prediction (ROC-AUC 0.844) using Scikit-Learn Pipeline. Covers data cleaning, EDA, feature engineering, stratified CV, tuning &amp; business-driven at-risk ranking. Top drivers: Month-to-month contract, low tenure, no OnlineSecurity.

# Telco Customer Churn Prediction

Predict which customers are likely to churn so the retention team can intervene proactively.

> End-to-end ML pipeline: Data Cleaning -> EDA -> Feature Engineering -> ColumnTransformer Pipeline -> Stratified Split -> CV Comparison -> Tuning -> Final Evaluation -> At-Risk Ranking

## 1. Business Problem
Telco churn rate is 26.5%. Acquiring new customers is 5x more expensive than retaining. Goal is to identify high-risk customers early.

## 2. Dataset
- 7043 customers, 21 features + target
- Source: Telco Churn (Kaggle / IBM)
- Target: `Churn` - Yes/No -> 1/0

## 3. Data Cleaning
- `TotalCharges` was stored as object because 11 rows had blank strings `" "`
- Fix: `pd.to_numeric(..., errors="coerce")` -> NaN
- Validation: `Rows with missing TotalCharges have tenure = [0]` - new customers, logical to fill as 0. Handled by median imputation in Pipeline.
- Dropped `customerID` (identifier), mapped `SeniorCitizen 0/1 -> No/Yes`, mapped `Churn -> 1/0`

## 4. EDA - Key Findings
1. **Contract is #1 driver:** Month-to-month churn ~42% vs One-year ~11% vs Two-year ~3%
2. **Fiber optic churns most:** Fiber ~42% vs DSL ~19% vs No internet ~7%
3. **New + High Spend = Risk:** Tenure 0-2 months churn cluster, Churned customers median MonthlyCharges ~$80 vs $65 for stayed
4. **Class Imbalance:** Stayed 5150 (73.5%) vs Churned 1870 (26.5%)

## 5. Feature Engineering
- `num_services`: Count of Yes across PhoneService, Internet, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- `avg_monthly_spend`: TotalCharges / tenure (with 0 tenure handled)
- `tenure_group`: Binned 0-1y, 1-2y, 2-4y, 4-6y using `pd.cut`
- Final `X` shape: (7043, 22)

## 6. Preprocessing Pipeline (No Data Leakage)
Wrapped in `ColumnTransformer` + `Pipeline` so fit only on training folds:

- **Numeric:** `['tenure','MonthlyCharges','TotalCharges','num_services','avg_monthly_spend']` -> `SimpleImputer(median)` + `StandardScaler()`
- **Categorical:** 17 cols -> `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`

## 7. Train-Test Split
`test_size=0.2, stratify=y, random_state=42` -> Train 5634 / Test 1409, churn rate preserved at 26.5% in both.

## 8. Model Comparison - Stratified 5-Fold CV (ROC-AUC)
`class_weight="balanced"` for imbalance.

| Model | Precision (Churn) | Recall (Churn) | F1 | ROC-AUC (CV) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.519 | 0.795 | 0.628 | **0.846** |
| Random Forest | 0.638 | 0.472 | 0.542 | 0.827 |

Logistic had better recall at baseline.

## 9. Hyperparameter Tuning
`GridSearchCV` on Random Forest, scoring `roc_auc`

Grid: `n_estimators [200,400], max_depth [6,10,None], min_samples_leaf [1,5]`
**Best Params:** `max_depth=6, min_samples_leaf=5, n_estimators=400`
**Best CV ROC-AUC: 0.848**

## 10. Final Evaluation - Held-out Test Set (1409)

| Model | Test ROC-AUC | Churn Precision | Churn Recall | Churn F1 | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.842 | 0.50 | 0.79 | 0.61 | 0.73 |
| **Random Forest (tuned)** | **0.844** | **0.52** | **0.80** | **0.63** | **0.75** |

Confusion Matrix (RF tuned): TN 761, FP 274, FN 76, TP 298
ROC curves show both models generalize well.

## 11. What Drives Churn? - Feature Importance (Random Forest)

Top 12:
1. `Contract_Month-to-month` (0.15)
2. `tenure`
3. `Contract_Two year`
4. `OnlineSecurity_No`
5. `TechSupport_No`
6. `TotalCharges`
7. `InternetService_Fiber optic`
8. `tenure_group_0-1y`
9. `MonthlyCharges`
10. `PaymentMethod_Electronic check`
11. `avg_monthly_spend`
12. `tenure_group_4-6y`

## 12. Business Impact - At-Risk Customers
Ranking test customers by `churn_probability`

**Result: Of the 50 highest-risk customers, 42 actually churned (84% precision in Top-50)**

Profile of top-risk: `tenure=1-3 months, Contract=Month-to-month, MonthlyCharges $76-96, prob 0.88-0.89`

Outputs saved:
- `outputs/top_at_risk_customers.csv` - ranked list for retention team
- `outputs/churn_model.joblib` - production model

## 13. Recommended Actions
1. Target 0-12 month, month-to-month, Fiber customers with 1-year discount + free OnlineSecurity/TechSupport bundle
2. Onboarding call at Day 30
3. Push AutoPay (avoid Electronic check)

## 14. Limitations & Next Steps
- Try XGBoost / LightGBM
- Tune decision threshold from 0.5 to 0.3 to optimize recall vs cost
- Add SHAP explainability
- Add cost-sensitive evaluation (retention cost vs LTV)
- Deployment via API

## 15. How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/telco_churn.ipynb
