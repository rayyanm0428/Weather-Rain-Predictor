"""
train_improved.py

Second iteration: a Random Forest with a couple of extra engineered
features (multi-day trends, not just single-day snapshots). Compared
directly against the baseline so we can honestly report whether the
added complexity actually helped.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from features import build_features, train_test_split_by_time, FEATURE_COLUMNS

DATA_PATH = Path(__file__).parent.parent / "data" / "weather_history.csv"
MODEL_PATH = Path(__file__).parent.parent / "models" / "improved_model.joblib"

EXTRA_FEATURES = ["humidity_3d_avg", "pressure_3d_trend"]
ALL_FEATURES = FEATURE_COLUMNS + EXTRA_FEATURES

def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["humidity_3d_avg"] = df["humidity_pct"].rolling(3).mean()
    df["pressure_3d_trend"] = df["pressure_hpa"].diff(3)
    return df.dropna(subset=EXTRA_FEATURES).reset_index(drop=True)

def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_features(raw)
    df = add_extra_features(df)
    train, test = train_test_split_by_time(df)

    X_train, y_train = train[ALL_FEATURES], train["rained_tomorrow"]
    X_test, y_test = test[ALL_FEATURES], test["rained_tomorrow"]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("=== Improved: Random Forest ===")
    print(f"Train days: {len(train)} | Test days: {len(test)}")
    print(f"Model accuracy: {accuracy_score(y_test, preds):.1%}")
    print(f"Model F1 score: {f1_score(y_test, preds):.3f}")
    print("\nConfusion matrix (rows=actual, cols=predicted, [no rain, rain]):")
    print(confusion_matrix(y_test, preds))
    print("\n" + classification_report(y_test, preds, target_names=["No rain", "Rain"]))

    print("Feature importances:")
    for feat, imp in sorted(zip(ALL_FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat:20s} {imp:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": ALL_FEATURES}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
