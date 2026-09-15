# Credit Risk Modeling

A machine learning project that predicts whether a LendingClub loan is likely to default.

## Project Objective

The goal of this project is to assess the creditworthiness of loan applicants and classify completed loans into:

- **0 — Fully Paid:** Non-default
- **1 — Charged Off:** Default

The project demonstrates how to handle imbalanced credit-risk data using SMOTE and explain model predictions using SHAP.

## Dataset

This project uses the LendingClub loan dataset.

The original dataset contains more than 2.2 million loan records and 145 columns. To improve efficiency, only the required features are loaded and completed loans are selected.

The raw `loan.csv` file is **not included in this repository** because of its large size.

## Project Workflow

1. Load the LendingClub dataset
2. Select relevant features
3. Filter completed loans
4. Create the binary default target
5. Clean and preprocess the data
6. Handle missing values
7. Split the data into training and testing sets
8. Apply SMOTE to the training data
9. Train multiple machine learning models
10. Evaluate model performance
11. Compare models
12. Explain predictions using SHAP
13. Generate visualizations

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

## Handling Imbalanced Data

The completed-loan dataset contains significantly more Fully Paid loans than Charged Off loans.

SMOTE (Synthetic Minority Over-sampling Technique) is applied **only to the training data**.

The test data remains untouched so that evaluation represents the original class distribution.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

### Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.634 | 0.307 | 0.653 | 0.418 | **0.702** |
| Decision Tree | 0.718 | 0.337 | 0.416 | 0.372 | 0.661 |
| Random Forest | **0.730** | **0.359** | 0.433 | 0.392 | 0.694 |

Based on ROC-AUC, **Logistic Regression** is the selected model.

## Explainability

SHAP (SHapley Additive exPlanations) is used to understand which features contribute to model predictions.

The project generates:

- SHAP summary plot
- SHAP feature importance
- Individual applicant SHAP explanation
- Random Forest feature importance

## Important Features

Important predictive features identified during the analysis include:

- Interest rate
- Loan term
- Loan grade
- Employment length
- Debt-to-income ratio
- Mortgage account history
- Verification status
- Home ownership
- Loan purpose
- Recent credit inquiries

## Exploratory Data Analysis

The project generates visualizations for:

- Default distribution
- Interest rate vs default
- Income vs default
- Default rate by loan grade
- Feature correlation

## Project Structure

```text
Credit-Risk-Model/
│
├── prediction.py
├── requirements.txt
├── README.md
└── .gitignore