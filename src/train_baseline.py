"""
train_baseline.py

The simplest reasonable model: logistic regression predicting whether it
rains tomorrow. This is the "safety net" result — get this working end to
end before trying anything fancier. Every later improvement gets compared
back to this number.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from features import build_features, train_test_split_by_time, FEATURE_COLUMNS

DATA_PATH = Path(__file__).parent.parent / "data" / "weather_history.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "baseline_model.joblib"

def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_features(raw)
    train, test = train_test_split_by_time(df)

    X_train, y_train = train[FEATURE_COLUMNS], train["rained_tomorrow"]
    X_test, y_test = test[FEATURE_COLUMNS], test["rained_tomorrow"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    # "Always guess the majority class" baseline - if our model can't beat
    # this, it isn't actually learning anything useful.
    majority_class = y_train.mode()[0]
    naive_preds = [majority_class] * len(y_test)
    naive_acc = accuracy_score(y_test, naive_preds)

    print("=== Baseline: Logistic Regression ===")
    print(f"Train days: {len(train)} | Test days: {len(test)}")
    print(f"Naive 'always guess majority class' accuracy: {naive_acc:.1%}")
    print(f"Model accuracy:                                {accuracy_score(y_test, preds):.1%}")
    print(f"Model F1 score:                                 {f1_score(y_test, preds):.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted, [no rain, rain]):")
    print(confusion_matrix(y_test, preds))
    print("\n" + classification_report(y_test, preds, target_names=["No rain", "Rain"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
