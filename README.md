# E-Commerce Customer Churn Prediction — CatBoost ML Model for TokoX

## Project Description
This project focuses on predicting customer churn in an Indonesian e-commerce company (TokoX).
High churn rates can significantly impact company revenue, so the goal is to build an accurate and explainable Machine Learning model to detect at-risk customers early.

Using CatBoostClassifier, this project identifies key churn drivers and provides actionable business recommendations to improve customer retention and reduce churn costs.

Key Highlights:
- Optimized CatBoost model with AUC = 0.975, Recall = 0.94, Precision = 0.87, and F2 = 0.923
- Optimal threshold = 0.40 to prioritize recall and detect early churn
- Clear data-driven recommendations for management, marketing, and engineering teams

## Business Context
TokoX is an e-commerce company facing increasing churn rates.
Losing existing customers increases acquisition costs and reduces profitability.

This analysis aims to:
1. Detect churn signals early using customer behavior, satisfaction, and engagement data.
2. Build a data-driven retention framework to improve marketing efficiency.
3. Provide predictive insights for CRM integration and automated re-engagement.

## Dataset Overview
| Attribute | Description |
|------------|--------------|
| Total Records | 5,630 customers |
| Target Variable | Churn (1 = churned, 0 = retained) |
| Data Type | Behavioral, demographic, and transactional |
| Source | Simulated dataset representing Indonesian e-commerce customer data |
| Split Ratio | 80:20 (train:test) |

## Key Features Description
| Feature | Description |
|----------|--------------|
| CustomerID | Unique customer identifier |
| Churn | Target flag: 1 if churned, 0 if retained |
| Tenure | Customer lifetime in months |
| PreferredLoginDevice | Most frequently used login device |
| CityTier | City classification (Tier 1–3 based on size and development) |
| WarehouseToHome | Distance from warehouse to customer’s home (km) |
| PreferredPaymentMode | Commonly used payment method |
| Gender | Customer gender |
| HourSpendOnApp | Average time spent on app (hours) |
| NumberOfDeviceRegistered | Number of registered devices |
| PreferedOrderCat | Most frequently ordered product category |
| SatisfactionScore | Customer satisfaction rating (1–5) |
| MaritalStatus | Customer marital status |
| NumberOfAddress | Number of registered addresses |
| Complain | Complaint status last month (1 = Yes, 0 = No) |
| OrderAmountHikeFromlastYear | Yearly spending increase (%) |
| CouponUsed | Number of coupons used |
| OrderCount | Total number of orders |
| DaySinceLastOrder | Days since last order |
| CashbackAmount | Average cashback received last month |

## Exploratory Data Analysis (EDA)
| Visualization | Description |
|----------------|--------------|
| Categorical Feature Distribution | Distribution of categorical variables across churn classes |
| Numerical Feature Distribution | Comparison of numerical features between churned and retained customers |
| Target/Label Distribution | Overall churn rate visualization |
| Feature Correlation Heatmap | Correlation and interaction between top features |
| ROC Curve | Visualization of model performance across classification thresholds |
| Precision-Recall Curve | Trade-off between recall and precision |
| Threshold vs Scores | Visualization of the optimal threshold (0.40) |
| Confusion Matrix | Breakdown of predicted vs actual churn labels |
| SHAP Summary Plot | Interpretability of feature importance and influence direction |

## Machine Learning Workflow
1. Data Preprocessing
   - Handle missing values, outliers, encoding, and scaling
   - Split data 80:20 for training and testing

2. Feature Engineering
   - Create derived metrics such as ComplainRate, OrderFrequency, and EngagementScore
   - Perform feature selection using correlation matrix and importance ranking

3. Model Development
   - Train multiple classifiers (Logistic Regression, Random Forest, XGBoost, CatBoost)
   - Perform hyperparameter tuning with grid search
   - Finalize CatBoost model for best balance between recall and precision

4. Model Evaluation
   - Evaluate models using ROC-AUC, Precision-Recall curve, F2-score, and Confusion Matrix
   - Determine optimal threshold (0.40) for operational deployment

5. Model Explainability
   - Apply SHAP analysis for feature interpretability
   - Identify key churn drivers: Tenure, ComplainRate, DaySinceLastOrder, and CashbackAmount

## Final Conclusions and Business Recommendations

### Summary
- Final Model: CatBoostClassifier
- Optimal Threshold: 0.40 (F2 = 0.923, Recall = 0.94, Precision = 0.87)
- AUC: 0.975 → Strong classification capability
- Accuracy: 97%

### Key Insights
Factors Increasing Churn Risk:
- Low Tenure → new customers with low loyalty
- High ComplainRate → frequent complaints signal dissatisfaction
- High DaySinceLastOrder → inactivity indicator
- Lower CityTier → limited logistics access and service quality

Factors Reducing Churn Risk:
- High CashbackAmount → loyalty incentive
- Frequent OrderFrequency → consistent engagement
- High SatisfactionScore → positive experience retention
- Grocery category preference → stable purchase routine

### Business Recommendations
1. Implement daily churn scoring with threshold 0.40 for early detection.
2. Use low-cost retention strategies (small vouchers, loyalty points, personalized reminders).
3. Prioritize high-value customers (based on LTV) for retention incentives.
4. Automate reminders and customer segmentation based on churn probability.

### Tactical Actions
| Segment | Characteristics | Recommended Action |
|----------|----------------|--------------------|
| High Risk | Low tenure, low activity, frequent complaints | Personalized push/email + small voucher |
| Medium Risk | Decreasing engagement | Product recommendation + engagement reminder |
| Low Risk | Active and loyal | Loyalty program, periodic cashback, appreciation messages |

### Data and Model Assumptions
- Dataset represents customer behavior without sampling bias.
- Churn defined as customers inactive for 90 days.
- No data leakage across train-test splits.
- Feature selection and preprocessing consistent across splits.
- Train-test split ratio = 80:20 for generalization.
- Threshold 0.40 chosen for optimal F2.
- Model evaluated using ROC-AUC, PR Curve, and SHAP for interpretability.

### Limitations
1. Generalization limited to one dataset snapshot — retraining required regularly.
2. Missing behavioral data (clickstream, CS feedback, marketing channel).
3. No integration of Customer Lifetime Value (LTV).
4. Predictions are probabilistic, not deterministic — should support, not replace, business judgment.

## Repository Structure
```
├── data/
│   └── ECommerce_Dataset.csv
├── notebook/
│   └── Final_Project_Beta_fix.ipynb
├── app/
│   └── streamlit_app.py
├── visuals/
│   └── (to be added manually)
├── model/
│   └── model_catboost.sav
├── README.md
```

## Author
**Gregorius Daniel**
**Imaculata Viandra**
Data Scientist | Purwadhika Bootcamp Graduate  
[LinkedIn Profile](https://www.linkedin.com/in/danieladityapr/)
