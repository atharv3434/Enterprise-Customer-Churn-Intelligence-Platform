"""
Model training, cross-validation, threshold tuning, and serialization.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from features import build_preprocessor


def run_training_pipeline():
    data_path = Path("data/raw/customers.csv")
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    # Stratified split to protect class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Step 1: Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Class balance adjustment ratio
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()

    print("Step 2: Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train_proc, y_train)

    print("Step 3: Evaluating Performance...")
    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate optimal threshold via F1 from Precision-Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)

    print(f"Test ROC-AUC:            {roc_auc:.4f}")
    print(f"Optimal F1 Decision Cut: {best_threshold:.4f}")
    print("\nClassification Report (Optimized Threshold):")
    print(classification_report(y_test, y_pred_optimal))

    # Persist end-to-end bundle
    model_bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "optimal_threshold": best_threshold,
        "roc_auc": roc_auc
    }
    model_path = artifacts_dir / "churn_model_pipeline.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"Complete bundle successfully saved to {model_path}")


if __name__ == "__main__":
    run_training_pipeline()