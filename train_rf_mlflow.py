import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report)
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING RANDOM FOREST WITH MLFLOW TRACKING")
print("="*70)

# Load and prepare data
print("\n[*] Loading data...")
SAMPLE_FRAC = 1_000_000 / 7_000_000
CHUNK_SIZE = 200_000
DATA_PATH = r'2009.csv/2009.csv'

chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df_raw = pd.concat(chunks, ignore_index=True)
print(f"[OK] Raw data loaded: {df_raw.shape}")

# Data preprocessing
print("\n[*] Preprocessing data...")

# Create target variable
df_raw['IS_DELAYED'] = (df_raw['ARR_DELAY'] >= 15).astype(int)

# Remove cancelled and diverted flights
df_clean = df_raw[(df_raw['CANCELLED'] == 0) & (df_raw['DIVERTED'] == 0)].copy()

# Drop columns with high missing values or leakage
drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
             'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

# Fill missing values with median
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Feature engineering
date_col = 'FL_DATE' if 'FL_DATE' in df_clean.columns else None
if date_col:
    df_clean['MONTH'] = pd.to_datetime(df_clean[date_col], errors='coerce').dt.month
    df_clean['DAY_OF_WEEK'] = pd.to_datetime(df_clean[date_col], errors='coerce').dt.dayofweek
else:
    df_clean['MONTH'] = np.random.randint(1, 13, len(df_clean))
    df_clean['DAY_OF_WEEK'] = np.random.randint(0, 7, len(df_clean))

# Extract hour from scheduled departure
if 'CRS_DEP_TIME' in df_clean.columns:
    df_clean['HOUR'] = df_clean['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
else:
    df_clean['HOUR'] = np.random.randint(0, 24, len(df_clean))

df_clean['IS_WEEKEND'] = (df_clean['DAY_OF_WEEK'] >= 5).astype(int)
df_clean['IS_RUSH_HOUR'] = ((df_clean['HOUR'] >= 7) & (df_clean['HOUR'] <= 9)).astype(int)

# Select features
feature_cols = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND', 'IS_RUSH_HOUR',
                'DEP_DELAY', 'DISTANCE']
feature_cols = [c for c in feature_cols if c in df_clean.columns]

X = df_clean[feature_cols].copy()
y = df_clean['IS_DELAYED'].copy()

print(f"[OK] Features: {feature_cols}")
print(f"[OK] Feature shape: {X.shape}")
print(f"[OK] Target distribution - Delayed: {y.mean()*100:.1f}%")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Train set: {X_train.shape[0]:,} samples")
print(f"[OK] Test set: {X_test.shape[0]:,} samples")

# MLflow configuration
print("\n[*] Setting up MLflow...")
project_root = os.getcwd()
mlflow_db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

experiment_name = "Flight_Delay_RF"

try:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f"[OK] Using existing experiment: {experiment_name}")
except:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"[OK] Created new experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)

# Train Random Forest with MLflow tracking
print("\n[*] Training Random Forest Classifier...")
print("-" * 70)

with mlflow.start_run(run_name="RF_n100_d10"):
    # Model parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }

    # Log parameters
    mlflow.log_params(model_params)

    # Train model
    rf_model = RandomForestClassifier(**model_params)
    rf_model.fit(X_train, y_train)

    print(f"[OK] Model training completed")

    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(rf_model, "model", input_example=X_test.iloc[:5])

    # Print results
    print("\n[METRICS] MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    print("\n[MATRIX] CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\nInterpretation:")
    print(f"  TN (True Negatives):  {cm[0,0]:,}")
    print(f"  FP (False Positives): {cm[0,1]:,}")
    print(f"  FN (False Negatives): {cm[1,0]:,}")
    print(f"  TP (True Positives):  {cm[1,1]:,}")

    # Classification report
    print("\n[REPORT] CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred,
                                target_names=['Not Delayed', 'Delayed']))

    # Save model to pickle file
    model_path = 'rf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"\n[OK] Model saved to: {model_path}")

    # MLflow info
    print("\n[INFO] MLflow Tracking Information")
    print("=" * 70)
    print(f"Experiment:  {experiment_name}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Run ID:      {mlflow.active_run().info.run_id}")
    print(f"\nTo view results, run: mlflow ui")
    print(f"Then open: http://localhost:5000")

print("\n" + "=" * 70)
print("[OK] TRAINING COMPLETE")
print("=" * 70)
