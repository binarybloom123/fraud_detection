# FRAUD DETECTION IN FINANCIAL TRANSACTIONS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# --------------------------------------------------
# STEP 1: LOAD HISTORICAL TRANSACTION DATA
# --------------------------------------------------

print("\nSTEP 1: LOADING DATASET")

data = pd.read_csv("creditcard.csv")

print("Original Dataset Shape:", data.shape)

print("Missing Values:", data.isnull().sum().sum())

print("Duplicate Records:", data.duplicated().sum())


# --------------------------------------------------
# STEP 2: DATA PREPROCESSING
# --------------------------------------------------

print("\nSTEP 2: DATA PREPROCESSING")

# Remove duplicate transactions
data = data.drop_duplicates()

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

print("Cleaned Dataset Shape:", data.shape)

print("\nTransaction Distribution:")
print(y.value_counts())

print("\nFraud Percentage:",
      round(y.mean() * 100, 3), "%")


# --------------------------------------------------
# STEP 3: FEATURE EXTRACTION
# --------------------------------------------------

print("\nSTEP 3: FEATURE EXTRACTION")

# Create a new feature from transaction amount
X["Amount_Log"] = np.log1p(X["Amount"])

# Create a time-based feature
# Time represents seconds elapsed in the dataset
X["Time_Hour_Proxy"] = (X["Time"] // 3600) % 24

print("New Features Created:")
print("1. Amount_Log")
print("2. Time_Hour_Proxy")


# --------------------------------------------------
# STEP 4: SPLIT DATASET
# --------------------------------------------------

print("\nSTEP 4: SPLITTING DATA")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# --------------------------------------------------
# STEP 5: FEATURE SELECTION
# --------------------------------------------------

print("\nSTEP 5: FEATURE SELECTION")

# Select the 10 most relevant features
selector = SelectKBest(
    score_func=f_classif,
    k=10
)

# Fit feature selection only on training data
X_train_selected = selector.fit_transform(X_train, y_train)

X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]

print("\nSelected Features:")
print(list(selected_features))


# --------------------------------------------------
# STEP 6: CLASSIFICATION MODELS
# --------------------------------------------------

print("\nSTEP 6: CLASSIFICATION MODELS")

# Create a preprocessing and training pipeline
def create_pipeline(model):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

        ("smote", SMOTE(random_state=42)),

        ("classifier", model)
    ])


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}


# Function to evaluate each model
def evaluate_model(name, model):

    print("\nTraining:", name)

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    y_prob = model.predict_proba(X_test_selected)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    pr_auc = average_precision_score(y_test, y_prob)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": pr_auc
    })

    trained_models[name] = model

    return y_pred


# Train classification models
for name, model in models.items():

    pipeline = create_pipeline(model)

    evaluate_model(name, pipeline)


# --------------------------------------------------
# STEP 7: ENSEMBLE LEARNING
# --------------------------------------------------

print("\nSTEP 7: ENSEMBLE LEARNING")

# Combine Logistic Regression and Random Forest
voting_model = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),

        ("rf", RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ))
    ],
    voting="soft"
)

voting_pipeline = create_pipeline(voting_model)

evaluate_model("Voting Classifier", voting_pipeline)


# --------------------------------------------------
# STEP 8: HYPERPARAMETER TUNING
# --------------------------------------------------

print("\nSTEP 8: HYPERPARAMETER TUNING")

# Use Random Forest for optimization
tuning_pipeline = create_pipeline(
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
)

parameter_grid = {
    "classifier__n_estimators": [50, 100],

    "classifier__max_depth": [10, 20, None],

    "classifier__min_samples_split": [2, 5]
}

search = RandomizedSearchCV(
    tuning_pipeline,
    parameter_grid,
    n_iter=3,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=1
)

# Use a smaller stratified training sample for faster tuning
tuning_X, _, tuning_y, _ = train_test_split(
    X_train_selected,
    y_train,
    train_size=0.25,
    random_state=42,
    stratify=y_train
)

search.fit(tuning_X, tuning_y)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)

# Evaluate optimized model
evaluate_model("Tuned Random Forest", best_model)


# --------------------------------------------------
# STEP 9: MODEL COMPARISON
# --------------------------------------------------

print("\nSTEP 9: MODEL COMPARISON")

results_df = pd.DataFrame(results)

print(results_df.round(4))

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# Visualize model performance
results_df.set_index("Model")[
    ["Precision", "Recall", "F1 Score", "PR-AUC"]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Fraud Detection Model Performance")
plt.ylabel("Score")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.close()


# --------------------------------------------------
# STEP 10: CONFUSION MATRIX
# --------------------------------------------------

print("\nSTEP 10: CONFUSION MATRIX")

y_pred = best_model.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Tuned Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.close()


# --------------------------------------------------
# STEP 11: MISCLASSIFIED TRANSACTIONS
# --------------------------------------------------

print("\nSTEP 11: MISCLASSIFIED TRANSACTIONS")

# Identify transactions predicted incorrectly
misclassified = X_test.copy()

misclassified["Actual_Class"] = y_test

misclassified["Predicted_Class"] = y_pred

misclassified = misclassified[
    misclassified["Actual_Class"] !=
    misclassified["Predicted_Class"]
]

print("Total Misclassified Transactions:",
      len(misclassified))

print("\nMisclassified Transactions:")
print(misclassified.head(10))

misclassified.to_csv(
    "misclassified_transactions.csv",
    index=False
)

# Visualize misclassified transactions
plt.figure(figsize=(8, 5))

sns.countplot(
    x="Actual_Class",
    hue="Predicted_Class",
    data=misclassified
)

plt.title("Misclassified Transactions")
plt.xlabel("Actual Class")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig("misclassified_transactions.png")
plt.close()


# --------------------------------------------------
# STEP 12: SAVE THE TRAINED MODEL
# --------------------------------------------------

print("\nSTEP 12: SAVING MODEL")

joblib.dump(
    {
        "model": best_model,
        "selected_features": list(selected_features)
    },
    "fraud_model.pkl"
)

print("Trained Model Saved Successfully!")


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("\nFRAUD DETECTION PROJECT COMPLETED")

print("All Models Trained and Evaluated")

print("Ensemble Learning Completed")

print("Hyperparameter Tuning Completed")

print("Visualizations and Results Saved")