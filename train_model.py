import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


def main():
    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    df = pd.read_csv("synthetic_fraud_dataset.csv")

    print("Original shape:", df.shape)

    # -----------------------------
    # 2. Data Cleaning
    # -----------------------------
    df = df.drop_duplicates()
    df = df.dropna()

    print("After cleaning:", df.shape)

    # -----------------------------
    # 3. Feature Engineering
    # -----------------------------
    df["Amount_Balance_Ratio"] = df["Transaction_Amount"] / (df["Account_Balance"] + 1)

    # -----------------------------
    # 4. Features & Target
    # -----------------------------
    feature_columns = [
        "Transaction_Amount",
        "Transaction_Type",
        "Account_Balance",
        "Device_Type",
        "Location",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Card_Type",
        "Transaction_Distance",
        "Authentication_Method",
        "Risk_Score",
        "Is_Weekend",
        "Amount_Balance_Ratio"   # new feature
    ]

    target_column = "Fraud_Label"

    X = df[feature_columns]
    y = df[target_column]

    # -----------------------------
    # 5. Define feature types
    # -----------------------------
    numeric_features = [
        "Transaction_Amount",
        "Account_Balance",
        "Previous_Fraudulent_Activity",
        "Daily_Transaction_Count",
        "Transaction_Distance",
        "Risk_Score",
        "Is_Weekend",
        "Amount_Balance_Ratio"
    ]

    categorical_features = [
        "Transaction_Type",
        "Device_Type",
        "Location",
        "Card_Type",
        "Authentication_Method"
    ]

    # -----------------------------
    # 6. Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # 7. Preprocessing
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse → dense (needed for SMOTE)
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    # -----------------------------
    # 8. Handle Imbalance (SMOTE)
    # -----------------------------
    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # -----------------------------
    # 9. Train Logistic Regression
    # -----------------------------
    model = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # -----------------------------
    # 10. Evaluation
    # -----------------------------
    y_train_pred = model.predict(X_train_resampled)
    y_test_pred = model.predict(X_test_processed)

    print("\nTrain Accuracy:", round(accuracy_score(y_train_resampled, y_train_pred), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # -----------------------------
    # 11. Save model
    # -----------------------------
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\nModel saved successfully ✅")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()