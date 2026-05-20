"""
src/train.py — Script principal d'entraînement (Tâche 5 — Partie 1)
Paramètres, métriques, artefacts (matrice de confusion + rapport) loggés dans MLflow.
Usage : python src/train.py [--retrain]
"""
import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, '2009.csv', '2009.csv')
MLFLOW_URI   = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT   = 'Flight_Delay_MLOps'
CHUNK_SIZE   = 200_000
SAMPLE_FRAC  = 1_000_000 / 7_000_000
RANDOM_STATE = 42

MODEL_PARAMS = {
    'model_type':    'RandomForest',
    'n_estimators':  100,
    'max_depth':     10,
    'test_size':     0.2,
    'random_state':  RANDOM_STATE,
    'class_weight':  'balanced',
}


def load_and_preprocess():
    print('[1/3] Chargement et prétraitement des données...')
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE))
    df = pd.concat(chunks, ignore_index=True)

    df['IS_DELAYED'] = (df['ARR_DELAY'] >= 15).astype(int)
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()

    drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
                 'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    if 'FL_DATE' in df.columns:
        df['MONTH']       = pd.to_datetime(df['FL_DATE'], errors='coerce').dt.month
        df['DAY_OF_WEEK'] = pd.to_datetime(df['FL_DATE'], errors='coerce').dt.dayofweek
    if 'CRS_DEP_TIME' in df.columns:
        df['HOUR'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)

    df['IS_WEEKEND']   = (df['DAY_OF_WEEK'] >= 5).astype(int)
    df['IS_RUSH_HOUR'] = ((df['HOUR'] >= 7) & (df['HOUR'] <= 9)).astype(int)

    features = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND',
                'IS_RUSH_HOUR', 'DEP_DELAY', 'DISTANCE']
    features = [c for c in features if c in df.columns]

    X = df[features]
    y = df['IS_DELAYED']
    print(f'    Dataset : {len(df):,} vols | Features : {features}')
    print(f'    Taux retards : {y.mean()*100:.1f}%')
    return X, y, features


def train_and_log(X, y, features, run_name='rf_baseline', retrain=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_PARAMS['test_size'],
        random_state=MODEL_PARAMS['random_state'], stratify=y
    )
    print(f'[2/3] Entraînement — run: {run_name} | Train={len(X_train):,} Test={len(X_test):,}')

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    tag = {'retrain': str(retrain), 'dataset': '2009_flights'}

    with mlflow.start_run(run_name=run_name, tags=tag) as run:
        # ── 1. Logger les paramètres ─────────────────────────────────────────
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param('n_features', len(features))
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size_n', len(X_test))

        # ── 2. Entraîner ─────────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=MODEL_PARAMS['n_estimators'],
            max_depth=MODEL_PARAMS['max_depth'],
            random_state=MODEL_PARAMS['random_state'],
            n_jobs=-1,
            class_weight=MODEL_PARAMS['class_weight'],
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ── 3. Logger les métriques ──────────────────────────────────────────
        metrics = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1_score':  f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        print(f'    Accuracy={metrics["accuracy"]:.4f}  F1={metrics["f1_score"]:.4f}  AUC={metrics["roc_auc"]:.4f}')

        # ── 4. Artefact : matrice de confusion ───────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=['A l\'heure', 'Retardé'],
            colorbar=False, ax=ax
        )
        ax.set_title(f'Matrice de confusion — {run_name}', fontweight='bold')
        plt.tight_layout()
        cm_path = os.path.join(PROJECT_ROOT, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path='artifacts')

        # ── 5. Artefact : rapport de classification ──────────────────────────
        report = classification_report(
            y_test, y_pred,
            target_names=['A l\'heure (0)', 'Retardé (1)']
        )
        report_path = os.path.join(PROJECT_ROOT, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f'Run : {run_name}\n')
            f.write(f'Experiment : {EXPERIMENT}\n\n')
            f.write(report)
        mlflow.log_artifact(report_path, artifact_path='artifacts')

        # ── 6. Logger le modèle ──────────────────────────────────────────────
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            registered_model_name='flight_delay_rf_production',
            input_example=X_test.iloc[:5],
        )

        run_id = run.info.run_id
        print(f'[3/3] Run ID : {run_id}')
        print(f'    Artefacts loggés : confusion_matrix.png, classification_report.txt, model/')
        return run_id, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Ré-entraînement déclenché par drift')
    args = parser.parse_args()

    run_name = 'rf_retrain' if args.retrain else 'rf_n100_d10_baseline'
    X, y, features = load_and_preprocess()
    run_id, metrics = train_and_log(X, y, features, run_name=run_name, retrain=args.retrain)

    print(f'\nTerminé. Run ID : {run_id}')
    print(f'Accuracy : {metrics["accuracy"]:.4f} | ROC-AUC : {metrics["roc_auc"]:.4f}')
    print(f'MLflow UI : mlflow ui --backend-store-uri sqlite:///{os.path.join(PROJECT_ROOT, "mlflow.db")} --port 5000')


if __name__ == '__main__':
    main()
