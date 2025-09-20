# Vodafone Customer Churn Prediction

## Project Overview
Customer churn (when subscribers leave a service) is a major challenge for telecom companies. Retaining existing customers is far cheaper than acquiring new ones.  
This project develops a predictive machine learning model to identify high-risk customers for Vodafone, enabling proactive retention strategies.

---

## Objectives
- Perform **Exploratory Data Analysis (EDA)** to understand churn patterns.  
- Build and evaluate **machine learning models** for churn prediction.  
- Identify **key drivers of churn** using explainability tools.  
- Deploy the final model via:  
  - Flask REST API  
  - Interactive HTML Form for quick demo  

---

## Dataset
- Source: Kaggle Vodafone Customer Churn dataset  
- Rows: 7,043  
- Columns: 23  
- Target: `Churn` (1 = Yes, 0 = No)  

Features include demographics, contract type, billing details, internet services, and support services.

---

## Exploratory Data Analysis (EDA)
Key insights from the dataset:

- **Class Imbalance**: 26% churned, 74% stayed → handled via SMOTE.  
- **Tenure Effect**: Shorter-tenure customers are more likely to churn.  
- **Charges**: Higher monthly charges linked to churn.  
- **Contract Type**: Month-to-month contracts show much higher churn than long-term ones.  
- **Support Services**: Lack of TechSupport and OnlineBackup increases churn risk.  

Visualizations included:
- Histograms & boxplots for numeric features (tenure, charges).  
- Correlation heatmap for numeric features.  
- Churn distribution plot showing imbalance.  

---

## Model Training & Evaluation
Three models were trained and compared:
1. Logistic Regression  
2. Random Forest  
3. XGBoost (final choice)  

**Evaluation Metrics**
- ROC-AUC: ~0.92–0.93 across all models.  
- Precision-Recall Curves: XGBoost performs best under imbalance.  
- Confusion Matrix & Classification Report: Balanced precision and recall after threshold tuning.  

**Final chosen model**: XGBoost with threshold tuned at 0.6.

---

## Results
- ROC-AUC: 0.926  
- Best Threshold (0.6):  
  - Precision = 0.72  
  - Recall = 0.86  
  - Balanced F1-score  

---

## Feature Importance & Explainability
Top drivers of churn identified by XGBoost:
- Contract Type (month-to-month vs long-term)  
- Payment Method (electronic check higher risk)  
- Tenure  
- Total & Monthly Charges  
- Support Services (TechSupport, OnlineBackup, etc.)  

SHAP analysis confirmed the importance of tenure, charges, and support services, providing both local and global explanations for predictions.

---

## Deployment
The final pipeline was exported as `churn_pipeline_artifacts.joblib` and deployed with Flask.

**Run the Flask API**
```bash
python app.py
```

---

## Findings & Recommendations (From My Perspective)

**Problem**  
Customer churn reduces revenue and customer lifetime value. Vodafone needs to identify at-risk customers and design retention strategies to minimize losses.

**Key Things to look at**
- **Contract Type**: Month-to-month customers churn at far higher rates than long-term ones.  
- **Tenure**: New customers (low tenure) are much more likely to churn.  
- **Payment Method**: Customers paying by electronic check show higher churn than those on auto-pay.  
- **Charges**: High monthly charges without perceived value increase churn risk.  
- **Support Services**: Lack of TechSupport, Online Security, or Backup correlates with higher churn.  

**Recommendations**
- Encourage longer-term contracts with discounts or loyalty perks.  
- Strengthen onboarding programs for new customers to reduce early churn.  
- Promote auto-pay options (credit card, bank transfer) to replace electronic checks.  
- Reassess pricing and offer bundled services to high-charge customers to boost value perception.  
- Drive adoption of support services with free trials or low-cost add-ons.  

**Business Impact**  
By focusing retention efforts on these high-risk groups, Vodafone can reduce churn, optimize marketing spend, and improve long-term profitability.

