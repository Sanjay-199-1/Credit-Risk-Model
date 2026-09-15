# ============================================================
# CREDIT RISK MODELING USING LENDINGCLUB DATASET
# OPTIMIZED VERSION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib

# IMPORTANT:
# Use a non-interactive backend so plt.show() does not pause
# the program in VS Code.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from imblearn.over_sampling import SMOTE

import shap

warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURATION
# ============================================================

FILE_PATH = r"C:\Users\Sanjay\OneDrive\Desktop\codec\loan.csv"

# Reduced sample size for faster processing.
# 100,000-150,000 is more than sufficient for this project.
SAMPLE_SIZE = 120000

RANDOM_STATE = 42

# Folder for project outputs
OUTPUT_DIR = "credit_risk_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. FEATURES TO LOAD
# ============================================================

# Only load columns required for the project.
#
# This is a major optimization because the original dataset
# contains 145 columns and more than 2.2 million rows.

selected_columns = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "loan_status"
]


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("=" * 60)
print("CREDIT RISK MODELING PROJECT")
print("=" * 60)

print("\nLoading LendingClub dataset...")
print("Only required columns are being loaded.")

try:

    df = pd.read_csv(
        FILE_PATH,
        usecols=selected_columns,
        low_memory=False
    )

except FileNotFoundError:

    print("\nERROR: Dataset file was not found.")
    print("Expected location:")
    print(FILE_PATH)
    raise


print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 5. LOAN STATUS DISTRIBUTION
# ============================================================

print("\nOriginal loan status distribution:")

print(
    df["loan_status"]
    .value_counts()
)


# ============================================================
# 6. KEEP ONLY COMPLETED LOANS
# ============================================================

# Fully Paid -> 0 -> Non-default
# Charged Off -> 1 -> Default
#
# Current / Late / Grace Period loans are excluded because
# their final outcome is not yet known.

df = df[
    df["loan_status"].isin(
        ["Fully Paid", "Charged Off"]
    )
].copy()


print("\nAfter selecting completed loans:")
print(df.shape)

print("\nCompleted loan distribution:")
print(
    df["loan_status"]
    .value_counts()
)


# ============================================================
# 7. CREATE TARGET VARIABLE
# ============================================================

df["default"] = (
    df["loan_status"] == "Charged Off"
).astype(int)

# Remove original loan_status
df.drop(
    columns=["loan_status"],
    inplace=True
)


print("\nTarget distribution:")
print(
    df["default"]
    .value_counts()
)

print("\nTarget percentage:")
print(
    df["default"]
    .value_counts(normalize=True) * 100
)


# ============================================================
# 8. SAMPLE DATASET
# ============================================================

# Instead of using 300,000 rows, we use 120,000.
#
# This is still a large enough dataset for:
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - SMOTE
# - SHAP
#
# It significantly reduces execution time.

sample_size = min(
    SAMPLE_SIZE,
    len(df)
)

df = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print("\nSampled dataset:")
print(df.shape)

print("\nSample target distribution:")
print(
    df["default"]
    .value_counts()
)


# ============================================================
# 9. DATA CLEANING
# ============================================================

print("\nCleaning data...")


# ------------------------------------------------------------
# Interest rate
# ------------------------------------------------------------

df["int_rate"] = (
    df["int_rate"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["int_rate"] = pd.to_numeric(
    df["int_rate"],
    errors="coerce"
)


# ------------------------------------------------------------
# Revolving utilization
# ------------------------------------------------------------

df["revol_util"] = (
    df["revol_util"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["revol_util"] = pd.to_numeric(
    df["revol_util"],
    errors="coerce"
)


# ------------------------------------------------------------
# Loan term
# ------------------------------------------------------------

df["term"] = (
    df["term"]
    .astype(str)
    .str.replace(" months", "", regex=False)
)

df["term"] = pd.to_numeric(
    df["term"],
    errors="coerce"
)


# ------------------------------------------------------------
# Employment length
# ------------------------------------------------------------

df["emp_length"] = (
    df["emp_length"]
    .astype(str)
    .replace({
        "< 1 year": "0",
        "10+ years": "10"
    })
    .str.extract(r"(\d+)", expand=False)
)

df["emp_length"] = pd.to_numeric(
    df["emp_length"],
    errors="coerce"
)


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

# Convert earliest credit line into credit history year.

df["earliest_cr_line"] = pd.to_datetime(
    df["earliest_cr_line"],
    format="%b-%Y",
    errors="coerce"
)

df["earliest_cr_year"] = (
    df["earliest_cr_line"]
    .dt.year
)

df.drop(
    columns=["earliest_cr_line"],
    inplace=True
)


# ============================================================
# 11. REMOVE INVALID VALUES
# ============================================================

# Annual income must be positive.

df = df[
    df["annual_inc"] > 0
].copy()


# DTI should be between 0 and 100.

df = df[
    (df["dti"] >= 0) &
    (df["dti"] <= 100)
].copy()


# ============================================================
# 12. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns.tolist()


# Numeric values -> median

for column in numeric_columns:

    if column != "default":

        df[column] = df[column].fillna(
            df[column].median()
        )


# Categorical values -> mode

for column in categorical_columns:

    if df[column].isna().any():

        mode_value = df[column].mode()

        if len(mode_value) > 0:

            df[column] = df[column].fillna(
                mode_value.iloc[0]
            )

        else:

            df[column] = df[column].fillna(
                "Unknown"
            )


print("\nTotal missing values after cleaning:")

print(
    df.isnull().sum().sum()
)


print("\nFinal cleaned dataset:")
print(df.shape)


# ============================================================
# 13. SAVE CLEANED DATASET
# ============================================================

cleaned_file = os.path.join(
    OUTPUT_DIR,
    "cleaned_credit_risk_data.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCleaned dataset saved to:")
print(cleaned_file)


# ============================================================
# 14. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nGenerating EDA graphs...")


# ------------------------------------------------------------
# Default distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="default",
    data=df
)

plt.title("Loan Default Distribution")
plt.xlabel("Default (0 = Fully Paid, 1 = Charged Off)")
plt.ylabel("Number of Loans")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_default_distribution.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Default percentage
# ------------------------------------------------------------

default_percentage = (
    df["default"]
    .value_counts(normalize=True)
    * 100
)

print("\nDefault percentage:")
print(default_percentage)


# ------------------------------------------------------------
# Interest rate vs default
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="default",
    y="int_rate",
    data=df
)

plt.title("Interest Rate vs Loan Default")
plt.xlabel("Default")
plt.ylabel("Interest Rate (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_interest_rate_vs_default.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Annual income vs default
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="default",
    y="annual_inc",
    data=df
)

# Limit extreme income values for visualization only.
plt.ylim(
    0,
    df["annual_inc"].quantile(0.99)
)

plt.title("Annual Income vs Loan Default")
plt.xlabel("Default")
plt.ylabel("Annual Income")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_income_vs_default.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Grade vs default
# ------------------------------------------------------------

grade_default = pd.crosstab(
    df["grade"],
    df["default"],
    normalize="index"
) * 100

print("\nDefault percentage by grade:")
print(grade_default)


grade_default.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Default Rate by Loan Grade")
plt.xlabel("Loan Grade")
plt.ylabel("Percentage")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_default_rate_by_grade.png"
    ),
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Correlation matrix
# ------------------------------------------------------------

numeric_for_corr = df.select_dtypes(
    include=np.number
)

correlation = numeric_for_corr.corr()


plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    center=0
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_correlation_matrix.png"
    ),
    dpi=150
)

plt.close()


print("EDA graphs saved.")


# ============================================================
# 15. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["default"]
)

y = df["default"]


# ============================================================
# 16. TRAIN TEST SPLIT
# ============================================================

# IMPORTANT:
# SMOTE is applied AFTER the train-test split.
#
# This prevents synthetic samples from leaking into the
# test set.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)


# ============================================================
# 17. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 18. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            StandardScaler(),
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        )

    ]
)


print("\nPreprocessing data...")


X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)


# ============================================================
# 19. APPLY SMOTE
# ============================================================

print("\n" + "=" * 60)
print("APPLYING SMOTE")
print("=" * 60)

print("\nBefore SMOTE:")

print(
    y_train.value_counts()
)


smote = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)


print("\nAfter SMOTE:")

print(
    y_train_smote.value_counts()
)


# ============================================================
# 20. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=500,
    solver="liblinear",
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_smote,
    y_train_smote
)

logistic_pred = logistic_model.predict(
    X_test_processed
)

logistic_prob = logistic_model.predict_proba(
    X_test_processed
)[:, 1]


print("Logistic Regression completed.")


# ============================================================
# 21. MODEL 2 — DECISION TREE
# ============================================================

print("\nTraining Decision Tree...")

decision_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=30,
    min_samples_leaf=10,
    random_state=RANDOM_STATE
)

decision_tree.fit(
    X_train_smote,
    y_train_smote
)

tree_pred = decision_tree.predict(
    X_test_processed
)

tree_prob = decision_tree.predict_proba(
    X_test_processed
)[:, 1]


print("Decision Tree completed.")


# ============================================================
# 22. MODEL 3 — RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

random_forest = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest.fit(
    X_train_smote,
    y_train_smote
)

rf_pred = random_forest.predict(
    X_test_processed
)

rf_prob = random_forest.predict_proba(
    X_test_processed
)[:, 1]


print("Random Forest completed.")


# ============================================================
# 23. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n" + "=" * 60)

    print(model_name)

    print("=" * 60)

    print(
        "Accuracy :", round(accuracy, 4)
    )

    print(
        "Precision:", round(precision, 4)
    )

    print(
        "Recall   :", round(recall, 4)
    )

    print(
        "F1 Score :", round(f1, 4)
    )

    print(
        "ROC-AUC  :", round(roc_auc, 4)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "Fully Paid",
                "Charged Off"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]


# ============================================================
# 24. EVALUATE MODELS
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

tree_results = evaluate_model(
    "Decision Tree",
    y_test,
    tree_pred,
    tree_prob
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# ============================================================
# 25. MODEL COMPARISON
# ============================================================

results = pd.DataFrame(
    [
        logistic_results,
        tree_results,
        rf_results
    ],
    columns=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    index=[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)


print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results)


# Save model results

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    )
)


# ============================================================
# 26. MODEL COMPARISON GRAPH
# ============================================================

results.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Credit Risk Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_model_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 27. ROC CURVES
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

tree_fpr, tree_tpr, _ = roc_curve(
    y_test,
    tree_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)


plt.figure(figsize=(9, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic Regression (AUC = {logistic_results[4]:.3f})"
)

plt.plot(
    tree_fpr,
    tree_tpr,
    label=f"Decision Tree (AUC = {tree_results[4]:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_results[4]:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves - Credit Risk Models")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_roc_curves.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 28. SELECT BEST MODEL
# ============================================================

best_model_name = results[
    "ROC-AUC"
].idxmax()


print("\nBest model based on ROC-AUC:")
print(best_model_name)


# ============================================================
# 29. GET BEST MODEL PREDICTIONS
# ============================================================

if best_model_name == "Logistic Regression":

    best_model = logistic_model
    best_predictions = logistic_pred
    best_probabilities = logistic_prob

elif best_model_name == "Decision Tree":

    best_model = decision_tree
    best_predictions = tree_pred
    best_probabilities = tree_prob

else:

    best_model = random_forest
    best_predictions = rf_pred
    best_probabilities = rf_prob


# ============================================================
# 30. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Fully Paid",
        "Charged Off"
    ],
    yticklabels=[
        "Fully Paid",
        "Charged Off"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_confusion_matrix.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# 31. FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


print("\nNumber of processed features:")
print(len(feature_names))


# ============================================================
# 32. SHAP EXPLAINABILITY
# ============================================================

print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

print("\nGenerating SHAP explanations...")


# ------------------------------------------------------------
# Use Random Forest for SHAP
# ------------------------------------------------------------

# We deliberately use a small sample.
#
# SHAP on the complete test dataset is unnecessary for an
# internship project and can take a long time.

shap_sample_size = min(
    300,
    X_test_processed.shape[0]
)


X_shap_sparse = X_test_processed[
    :shap_sample_size
]


# TreeExplainer requires manageable input for this step.

if hasattr(
    X_shap_sparse,
    "toarray"
):

    X_shap = X_shap_sparse.toarray()

else:

    X_shap = X_shap_sparse


print(
    f"Calculating SHAP values for {shap_sample_size} applicants..."
)


try:

    explainer = shap.TreeExplainer(
        random_forest
    )

    shap_values = explainer.shap_values(
        X_shap
    )

    # --------------------------------------------------------
    # Handle SHAP output versions
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        # Binary classification
        shap_values_default = shap_values[1]

    elif len(shap_values.shape) == 3:

        # Some newer SHAP versions
        shap_values_default = shap_values[:, :, 1]

    else:

        shap_values_default = shap_values


    # ========================================================
    # 33. SHAP SUMMARY PLOT
    # ========================================================

    plt.figure()

    shap.summary_plot(
        shap_values_default,
        X_shap,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "09_shap_summary.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 34. SHAP BAR PLOT
    # ========================================================

    plt.figure()

    shap.summary_plot(
        shap_values_default,
        X_shap,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "10_shap_feature_importance.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # 35. INDIVIDUAL APPLICANT EXPLANATION
    # ========================================================

    applicant_index = 0

    plt.figure()

    shap.force_plot(
        explainer.expected_value[1]
        if isinstance(
            explainer.expected_value,
            np.ndarray
        )
        else explainer.expected_value,

        shap_values_default[
            applicant_index
        ],

        X_shap[
            applicant_index
        ],

        feature_names=feature_names,

        matplotlib=True,

        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "11_individual_applicant_shap.png"
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print("SHAP analysis completed successfully.")


except Exception as shap_error:

    print("\nWARNING: SHAP analysis could not be completed.")

    print(
        "SHAP error:",
        shap_error
    )

    print(
        "\nThe predictive models were still trained successfully."
    )


# ============================================================
# 36. FEATURE IMPORTANCE — RANDOM FOREST
# ============================================================

print("\nGenerating Random Forest feature importance...")


try:

    importances = random_forest.feature_importances_

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(20)
    )


    print("\nTop 20 Random Forest features:")

    print(
        feature_importance
    )


    plt.figure(
        figsize=(10, 7)
    )

    sns.barplot(
        data=feature_importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Top 20 Random Forest Features"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "12_random_forest_feature_importance.png"
        ),
        dpi=150
    )

    plt.close()


except Exception as importance_error:

    print(
        "Feature importance error:",
        importance_error
    )


# ============================================================
# 37. SAVE FINAL PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["actual_default"] = (
    y_test.values
)

prediction_output["predicted_default"] = (
    best_predictions
)

prediction_output["default_probability"] = (
    best_probabilities
)

prediction_output["risk_level"] = np.where(
    best_probabilities >= 0.70,
    "High Risk",
    np.where(
        best_probabilities >= 0.40,
        "Medium Risk",
        "Low Risk"
    )
)


prediction_file = os.path.join(
    OUTPUT_DIR,
    "credit_risk_predictions.csv"
)

prediction_output.to_csv(
    prediction_file,
    index=False
)


print("\nPredictions saved to:")
print(prediction_file)


# ============================================================
# 38. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CREDIT RISK MODELING PROJECT COMPLETED")
print("=" * 60)

print("\nFinal dataset size:")
print(df.shape)

print("\nTraining samples before SMOTE:")
print(len(y_train))

print("\nTraining samples after SMOTE:")
print(len(y_train_smote))

print("\nTesting samples:")
print(len(y_test))

print("\nBest model:")
print(best_model_name)

print("\nModel results:")
print(results)

print("\nTarget definition:")
print("0 = Fully Paid / Non-default")
print("1 = Charged Off / Default")

print("\nSMOTE:")
print("Applied only to training data.")

print("\nExplainability:")
print("SHAP used for model interpretation.")

print("\nOutput directory:")
print(os.path.abspath(OUTPUT_DIR))

print("\nFiles generated:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        " -",
        filename
    )

print("\nProject completed successfully.")