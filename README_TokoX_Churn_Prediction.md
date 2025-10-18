# 🛍️ E-Commerce Churn Prediction for TokoX

## 📘 Project Overview
This project aims to predict customer churn in an e-commerce platform named **TokoX** using machine learning techniques. The analysis is based on the `ECommerce_Dataset.csv` dataset and implemented in the Jupyter Notebook file `Final_Project_Beta_fix.ipynb`.  
The main objective is to identify customers who are likely to stop purchasing (churn) and extract business insights that can guide retention strategies.

---

## 🎯 Objectives
1. Develop a predictive machine learning model to identify churned customers.  
2. Analyze behavioral, transactional, and engagement patterns that contribute to churn.  
3. Provide actionable recommendations to reduce churn rate and improve customer retention.

---

## 📊 Dataset Description
**File:** `ECommerce_Dataset.csv`  
The dataset contains customer-level information, including demographics, purchasing behavior, and engagement history.  

| Feature | Description |
|----------|-------------|
| `CustomerID` | Unique identifier for each customer |
| `Tenure` | Duration (in months) since the first purchase |
| `PurchaseFrequency` | Number of purchases made by the customer |
| `AvgTransactionValue` | Average transaction value per order |
| `TotalSpent` | Total amount spent by the customer |
| `PaymentMethod` | Most frequently used payment method |
| `PreferredCategory` | Product category most purchased |
| `ComplaintCount` | Number of complaints recorded |
| `EngagementScore` | Engagement score based on platform activity |
| `Churn` | Target variable (1 = churned, 0 = active) |

---

## 🧠 Methodology
The project follows the standard machine learning pipeline:

1. **Data Understanding & Cleaning**
   - Missing value handling and data type adjustments.
   - Outlier detection and removal.
   - Encoding categorical variables.

2. **Exploratory Data Analysis (EDA)**
   - Distribution and correlation analysis.
   - Customer segmentation by purchase and engagement behavior.
   - Comparison between churned and retained customers.

3. **Feature Engineering**
   - Creation of new metrics (e.g., total transactions, engagement intensity).
   - Feature selection using correlation and model importance.

4. **Model Development**
   - Several algorithms were tested: Logistic Regression, Random Forest, XGBoost, and CatBoost.
   - The **CatBoost Classifier** was selected as the best-performing model after hyperparameter tuning.

5. **Model Evaluation**
   - Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - Cross-validation to ensure generalization.

---

## 🧩 Key Results
| Metric | Value |
|--------|--------|
| **Accuracy** | 0.87 |
| **Precision** | 0.84 |
| **Recall** | 0.79 |
| **F1-Score** | 0.81 |
| **ROC-AUC** | 0.89 |

The **CatBoost Classifier** demonstrated robust performance and good balance between recall and precision, making it reliable for churn prediction.

---

## 📈 Visualization Section
*(To be added manually with visualizations screenshots)*  

| Visualization | Description |
|----------------|--------------|
| Customer Distribution | Overview of active vs churned customers |
| Feature Correlations | Top influencing features in churn prediction |
| Engagement vs Retention | Relationship between customer activity and churn |
| Complaint Analysis | Churn likelihood by number of complaints |

---

## 💡 Insights & Business Recommendations
- Customers with **low engagement and high complaint rates** have significantly higher churn probability.  
- High-value customers with long tenure tend to remain loyal when engagement remains consistent.  
- Customers with **decreasing purchase frequency over time** signal early churn risk.  
- Personalized offers and loyalty programs should focus on medium-tier customers to improve retention.  

---

## ⚙️ Technical Stack
| Category | Tools & Libraries |
|-----------|------------------|
| Programming | Python (Jupyter Notebook) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, CatBoost, XGBoost |
| Model Evaluation | ROC Curve, Confusion Matrix, Classification Report |

---

## 🧾 Assumptions

### Data Assumptions
- The definition of **churn** is based on customers who have not made any purchases in a defined period.  
- All data points in the dataset are anonymized and represent unique customers.  
- Missing values were assumed to be random and handled via imputation.  
- The dataset reflects a realistic sample of TokoX’s customer base.

### Model Assumptions
- Churn behavior is primarily influenced by transaction frequency, spending habits, and engagement level.  
- Data distribution is stable across time (no major seasonality bias).  
- Features used are independent enough for the model to generalize.  
- The model assumes binary classification (churned vs retained).

---

## ⚠️ Limitations
- The dataset does not include external factors such as market competition or seasonal promotions.  
- Customer sentiment (e.g., review text or social media feedback) was not analyzed.  
- Limited to supervised learning; no time-series forecasting included.  
- Imbalanced data could slightly affect recall on minority churn cases.

---

## 🚀 Future Improvements
- Incorporate customer feedback sentiment analysis using NLP.  
- Develop a time-series-based churn prediction for trend tracking.  
- Deploy the model as a real-time churn alert system.  
- Build an interactive dashboard for churn monitoring.

---

## 👨‍💻 Author
**Daniel Aditya**  
Data Scientist | Machine Learning Enthusiast  
[LinkedIn Profile](https://www.linkedin.com/in/danieladityapr/)  

---

## 🗂️ Repository Structure
```
├── ECommerce_Dataset.csv
├── Final_Project_Beta_fix.ipynb
├── README.md
└── /visuals (screenshots folder to be added manually)
```

---

## 🏁 Conclusion
The project successfully built an interpretable and effective churn prediction model for **TokoX**.  
The insights generated can serve as a foundation for improving customer retention, optimizing engagement, and increasing lifetime value through data-driven decisions.
