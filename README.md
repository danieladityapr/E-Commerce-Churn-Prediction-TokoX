## Final Project JCDSOH02-001

Beta Team :

- Gregorius Daniel Aditya Pratama
- Immaculata Viandra

# E-Commerce Customer Churn Prediction — CatBoost ML Model for TokoX

## Project Description
This project focuses on predicting customer churn using ECommerce_Dataset.csv in an Indonesian e-commerce company (TokoX).
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
   - Create derived metrics such as ComplainRate & OrderFrequency
   - Perform feature selection using correlation matrix and importance ranking

3. Model Development
   - Train multiple classifiers (Logistic Regression, Random Forest, XGBoost, CatBoost)
   - Perform hyperparameter tuning with grid search
   - Finalize CatBoost model for best balance F2_Score

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
Factors Increasing Churn Risk
- High ComplainRate → the more complaints, the higher the likelihood of churn.
- Low DaySinceLastOrder → customers recently made a transaction.
- High OrderFrequency → frequent transactions may indicate churn risk.
- High SatisfactionScore → positive experiences may raise customer expectations excessively.
- Low CityTier → customers in areas with limited logistics are more likely to churn.

Factors Decreasing Churn Risk
- High Tenure → long-term customers tend to be more loyal.
- High CashbackAmount → incentives increase loyalty.
- Active in app & grocery transactions → customers regularly use core services.

### Business Recommendations
- Customer Service (CS)
   - Fast Response Team (<24h) with chatbot + escalation.
   - Ticket system: Received → In Progress → Resolved → Verified.
   - Instant compensation (USD 3–5 voucher / free shipping).

- Marketing & CRM
   - Early Intervention for new customers (0–3 months).
   - Personalized campaigns: thank-you email + voucher, reminder after 7 days inactive.
   - Transactional Churn Alert: survey + auto-compensation if complaint detected.

- Data Analytics / BI
   - Build Customer Health Score (CHS) = satisfaction + complaints + frequency.
   - Daily CHS dashboard for CS & Marketing.
   - Weekly churn prediction report for Management.

- Operations & Logistics
   - Ensure stock for fast-moving items.
   - Priority delivery for “At Risk” customers.
   - SLA: 95% on-time delivery; auto-compensation for delays.

- Top Management
   - Set churn OKR (e.g., -5% in 6 months).
   - Allocate retention budget (USD 12k/year).
   - Quarterly review + cross-division KPI on retention.

### Data and Model Assumptions
- Dataset represents customer behavior without sampling bias.
- Churn defined as customers inactive for 90 days.
- No data leakage across train-test splits.
- Feature selection and preprocessing consistent across splits.
- Train-test split ratio = 80:20 for generalization.
- Using Final model = CatboostClassifier
- Threshold 0.40 chosen for optimal F2.
- Model evaluated using ROC-AUC, PR Curve, and SHAP for interpretability.

### Limitations
- Temporal Drift: The model is built on historical snapshots; churn patterns may shift due to seasonal promotions, pricing changes, or competitor moves. Without regular retraining, performance may degrade.
- Missing Features: Some relevant variables are not included (e.g., clickstream/behavioral data, acquisition channel, CS logs, specific promotions), which may omit key churn drivers.
- No CLV Integration: The model does not account for Customer Lifetime Value, so churn predictions do not prioritize retention efforts based on economic impact.
- Probabilistic Trade-off: Predictions are probabilities — false positives/negatives will occur. Retention actions (e.g., vouchers) must balance cost vs. expected benefit.
- Scalability of Personal Outreach: Personalized outreach works for small at-risk segments but requires automation when the segment grows, to keep costs sustainable.

### Mitigation Recommendations
- Retraining Cadence: Retrain every 1–3 months (or after major KPI shifts) with ongoing daily/weekly monitoring.
- Data Integration: Add clickstream, CS logs, and acquisition channel data to improve accuracy and interpretability.
- CLV-Based Segmentation: Incorporate CLV to prioritize interventions for high-value customers.
- A/B Testing: Test retention actions (vouchers, free shipping, outreach) to measure ROI and avoid overspending.
- Operational Decision Rules: Combine threshold + business rules (e.g., CHS) to guide intervention levels (low-cost automation vs. high-touch outreach).

## Repository Structure
```
├── data/
│   └── ECommerce_Dataset.csv
├── notebook/
│   └── Final_Project_Beta_fix.ipynb
├── app/
│   └── streamlit_app.py
├── model/
│   └── model_catboost.sav
├── README.md
```

## Author
| **Gregorius Daniel** |
| **Immaculata Viandra** |
Data Scientist | Purwadhika Bootcamp Graduate |
