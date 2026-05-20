"""
src/compare_models.py — Partie 2 : Comparaison de 6 algorithmes
Runs : RF baseline, RF profond, GradientBoosting, LogisticRegression, AdaBoost, XGBoost
Identifie programmatiquement le meilleur run par accuracy.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, '2009.csv', '2009.csv')
MLFLOW_URI   = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT   = 'Flight_Delay_MLOps'
RANDOM_STATE = 42

CONFIGS = [
    {
        'run_name':    'rf_baseline_n50_d3',
        'model_type':  'RandomForest',
        'model':       RandomForestClassifier(n_estimators=50,  max_depth=3,  random_state=RANDOM_STATE, n_jobs=-1),
        'params':      {'model_type': 'RandomForest', 'n_estimators': 50, 'max_depth': 3},
    },
    {
        'run_name':    'rf_deep_n200_d10',
        'model_type':  'RandomForest',
        'model':       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        'params':      {'model_type': 'RandomForest', 'n_estimators': 200, 'max_depth': 10},
    },
    {
        'run_name':    'gb_n100_lr01',
        'model_type':  'GradientBoosting',
        'model':       GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE),
        'params':      {'model_type': 'GradientBoosting', 'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
    },
    {
        'run_name':    'logreg_C1',
        'model_type':  'LogisticRegression',
        'model':       LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        'params':      {'model_type': 'LogisticRegression', 'C': 1.0, 'max_iter': 1000},
    },
    {
        'run_name':    'adaboost_n100',
        'model_type':  'AdaBoost',
        'model':       AdaBoostClassifier(
                           estimator=DecisionTreeClassifier(max_depth=3),
                           n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE
                       ),
        'params':      {'model_type': 'AdaBoost', 'n_estimators': 100, 'learning_rate': 0.5, 'base_depth': 3},
    },
    {
        'run_name':    'xgboost_n100',
        'model_type':  'XGBoost',
        'model':       xgb.XGBClassifier(
                           n_estimators=100, max_depth=5, learning_rate=0.1,
                           use_label_encoder=False, eval_metric='logloss',
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
                       ),
        'params':      {'model_type': 'XGBoost', 'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
    },
]


def load_data():
    print('[*] Chargement des données...')
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200_000, low_memory=False):
        chunks.append(chunk.sample(frac=1_000_000/7_000_000, random_state=RANDOM_STATE))
    df = pd.concat(chunks, ignore_index=True)

    df['IS_DELAYED'] = (df['ARR_DELAY'] >= 15).astype(int)
    df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)].copy()
    drop_cols = ['ARR_TIME','ARR_DELAY','ACTUAL_ELAPSED_TIME','AIR_TIME',
                 'WHEELS_OFF','WHEELS_ON','CANCELLED','DIVERTED','TAIL_NUM']
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

    features = [c for c in ['MONTH','DAY_OF_WEEK','HOUR','IS_WEEKEND','IS_RUSH_HOUR','DEP_DELAY','DISTANCE'] if c in df.columns]
    return df[features], df['IS_DELAYED']


def run_comparison():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f'Train={len(X_train):,} | Test={len(X_test):,}\n')

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    results = []

    for cfg in CONFIGS:
        print(f'[RUN] {cfg["run_name"]} ...')
        with mlflow.start_run(run_name=cfg['run_name']) as run:
            mlflow.log_params({**cfg['params'], 'test_size': 0.2, 'random_state': RANDOM_STATE})

            cfg['model'].fit(X_train, y_train)
            y_pred  = cfg['model'].predict(X_test)
            y_proba = cfg['model'].predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy':  accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall':    recall_score(y_test, y_pred, zero_division=0),
                'f1_score':  f1_score(y_test, y_pred, zero_division=0),
                'roc_auc':   roc_auc_score(y_test, y_proba),
            }
            mlflow.log_metrics(metrics)

            # Matrice de confusion pour chaque run
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=["A l'heure", 'Retardé'],
                colorbar=False, ax=ax
            )
            ax.set_title(cfg['run_name'], fontweight='bold')
            plt.tight_layout()
            cm_path = os.path.join(PROJECT_ROOT, f'cm_{cfg["run_name"]}.png')
            plt.savefig(cm_path, dpi=120, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(cm_path, artifact_path='artifacts')
            os.remove(cm_path)

            mlflow.sklearn.log_model(cfg['model'], 'model', input_example=X_test.iloc[:5])

            results.append({'run_id': run.info.run_id, 'run_name': cfg['run_name'], **metrics})
            print(f'  Accuracy={metrics["accuracy"]:.4f}  F1={metrics["f1_score"]:.4f}  AUC={metrics["roc_auc"]:.4f}')

    # ── Tableau comparatif ────────────────────────────────────────────────────
    df_res = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    print('\n' + '='*75)
    print('  TABLEAU COMPARATIF — 6 ALGORITHMES')
    print('='*75)
    print(f'  {"Run":<28} {"Accuracy":>10} {"F1":>8} {"Recall":>8} {"AUC":>8}')
    print('  ' + '-'*67)
    for _, row in df_res.iterrows():
        print(f'  {row["run_name"]:<28} {row["accuracy"]:>10.4f} {row["f1_score"]:>8.4f} {row["recall"]:>8.4f} {row["roc_auc"]:>8.4f}')

    # ── Requête programmatique — meilleur run ─────────────────────────────────
    print('\n' + '='*75)
    print('  MEILLEUR RUN (requête programmatique MlflowClient)')
    print('='*75)
    client     = MlflowClient(tracking_uri=MLFLOW_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    runs       = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['metrics.accuracy DESC'],
        max_results=5
    )
    best = runs[0]
    print(f'  Meilleur run  : {best.info.run_name}')
    print(f'  Run ID        : {best.info.run_id}')
    print(f'  Accuracy      : {best.data.metrics.get("accuracy", 0):.4f}')
    print(f'  F1-score      : {best.data.metrics.get("f1_score", 0):.4f}')
    print(f'  ROC-AUC       : {best.data.metrics.get("roc_auc", 0):.4f}')
    print(f'  Paramètres    : {best.data.params}')

    return best.info.run_id


if __name__ == '__main__':
    best_run_id = run_comparison()
    print(f'\n[OK] Comparaison terminée. Meilleur run ID : {best_run_id}')
    print(f'Lancez maintenant : python src/register_best_model.py')
