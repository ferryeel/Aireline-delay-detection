"""
Train a RandomForestClassifier, plot feature importances, and log to MLflow.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

try:
    from config import MLFLOW_TRACKING_URI
except Exception:
    MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "2009.csv" / "2009.csv"

EXPERIMENT_NAME = "Flight_Delay_RF"
RANDOM_STATE = 42
CHUNK_SIZE = 200_000
SAMPLE_FRAC = 1_000_000 / 7_000_000


def load_data():
    usecols = [
        "FL_DATE",
        "OP_CARRIER",
        "ORIGIN",
        "CRS_DEP_TIME",
        "DEP_DELAY",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "WEATHER_DELAY",
        "ARR_DELAY",
        "CANCELLED",
        "DIVERTED",
    ]

    chunks = []
    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE))

    df_raw = pd.concat(chunks, ignore_index=True)

    df_raw["IS_DELAYED"] = (df_raw["ARR_DELAY"] >= 15).astype(int)
    df_clean = df_raw[(df_raw["CANCELLED"] == 0) & (df_raw["DIVERTED"] == 0)].copy()

    df_clean["MONTH"] = pd.to_datetime(df_clean["FL_DATE"], errors="coerce").dt.month
    df_clean["DAY_OF_WEEK"] = pd.to_datetime(df_clean["FL_DATE"], errors="coerce").dt.dayofweek
    df_clean["HOUR"] = (
        df_clean["CRS_DEP_TIME"].astype(str).str.zfill(4).str[:2].astype(int)
    )
    df_clean["IS_WEEKEND"] = (df_clean["DAY_OF_WEEK"] >= 5).astype(int)
    df_clean["IS_RUSH_HOUR"] = (
        (df_clean["HOUR"] >= 7) & (df_clean["HOUR"] <= 9)
    ).astype(int)

    numeric_features = [
        "HOUR",
        "MONTH",
        "DAY_OF_WEEK",
        "IS_WEEKEND",
        "IS_RUSH_HOUR",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "DEP_DELAY",
        "WEATHER_DELAY",
    ]
    categorical_features = ["ORIGIN", "OP_CARRIER"]

    for col in numeric_features:
        if col not in df_clean.columns:
            continue
        if col == "WEATHER_DELAY":
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in categorical_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("UNKNOWN").astype(str)

    feature_cols = [c for c in numeric_features if c in df_clean.columns]
    categorical_cols = [c for c in categorical_features if c in df_clean.columns]

    X = df_clean[feature_cols + categorical_cols].copy()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    y = df_clean["IS_DELAYED"].copy()

    return X, y


def main():
    print("[INFO] Loading data...")
    X, y = load_data()

    print(f"[INFO] Dataset ready: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_params = {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name="RF_feature_importance"):
        mlflow.log_params(model_params)

        print("[INFO] Training RandomForestClassifier...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {"Variable": X.columns, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        top_3 = importance_df.head(3)
        top_features = top_3["Variable"].tolist()

        mlflow.log_params(
            {
                "top_feature_1": top_features[0],
                "top_feature_2": top_features[1],
                "top_feature_3": top_features[2],
            }
        )

        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "feature_importance_rf.csv"
        importance_df.to_csv(csv_path, index=False)

        fig_height = max(6, 0.2 * len(importance_df))
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.barh(importance_df["Variable"], importance_df["Importance"], color="#2f6f8f")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_ylabel("Variable")
        ax.set_title("Importance des variables — Random Forest")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout()

        plot_path = output_dir / "feature_importance_rf.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(plot_path), artifact_path="feature_importance")
        mlflow.log_artifact(str(csv_path), artifact_path="feature_importance")
        mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])

        print("[OK] Top 3 features:")
        for idx, row in enumerate(top_3.itertuples(index=False), 1):
            print(f"  {idx}. {row.Variable} -> {row.Importance:.6f}")

        print(f"[OK] Plot saved to: {plot_path}")
        print(f"[OK] CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
