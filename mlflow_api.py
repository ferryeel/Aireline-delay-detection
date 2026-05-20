"""
Flask API to retrieve MLflow experiment data and serve it to the frontend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import mlflow
import mlflow.sklearn
import os
import json
import pickle
import threading
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

app = Flask(__name__)
CORS(app)

# Configure MLflow
project_root = os.path.dirname(os.path.abspath(__file__))
mlflow_db_path = os.path.join(project_root, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

DEFAULT_EXPERIMENT = "airline_delay_task3"
DEFAULT_SAMPLE_FRAC = 1_000_000 / 7_000_000
CHUNK_SIZE = 200_000
RANDOM_STATE = 42

MODEL_SAMPLE_FRAC = {
    "random-forest": DEFAULT_SAMPLE_FRAC,
    "svm": 300_000 / 7_000_000,
    "xgboost": 500_000 / 7_000_000,
    "adaboost": 500_000 / 7_000_000
}

MODEL_LABELS = {
    "random-forest": "Random Forest",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "adaboost": "AdaBoost"
}

TRAINING_STATE = {
    "random-forest": {"status": "idle", "run_id": None, "metrics": None, "params": None, "experiment_id": None, "error": None},
    "svm": {"status": "idle", "run_id": None, "metrics": None, "params": None, "experiment_id": None, "error": None},
    "xgboost": {"status": "idle", "run_id": None, "metrics": None, "params": None, "experiment_id": None, "error": None},
    "adaboost": {"status": "idle", "run_id": None, "metrics": None, "params": None, "experiment_id": None, "error": None}
}


def load_training_data(data_path, sample_frac, chunk_size, random_state=42):
    """Load and preprocess flight delay data for training."""
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
        if sample_frac < 1:
            chunk = chunk.sample(frac=sample_frac, random_state=random_state)
        chunks.append(chunk)

    df_raw = pd.concat(chunks, ignore_index=True)
    df_raw["IS_DELAYED"] = (df_raw["ARR_DELAY"] >= 15).astype(int)

    df_clean = df_raw[(df_raw["CANCELLED"] == 0) & (df_raw["DIVERTED"] == 0)].copy()

    drop_cols = [
        "ARR_TIME", "ARR_DELAY", "ACTUAL_ELAPSED_TIME", "AIR_TIME",
        "WHEELS_OFF", "WHEELS_ON", "CANCELLED", "DIVERTED", "TAIL_NUM"
    ]
    df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    date_col = "FL_DATE" if "FL_DATE" in df_clean.columns else None
    if date_col:
        df_clean["MONTH"] = pd.to_datetime(df_clean[date_col], errors="coerce").dt.month
        df_clean["DAY_OF_WEEK"] = pd.to_datetime(df_clean[date_col], errors="coerce").dt.dayofweek
    else:
        df_clean["MONTH"] = np.random.randint(1, 13, len(df_clean))
        df_clean["DAY_OF_WEEK"] = np.random.randint(0, 7, len(df_clean))

    if "CRS_DEP_TIME" in df_clean.columns:
        df_clean["HOUR"] = df_clean["CRS_DEP_TIME"].astype(str).str.zfill(4).str[:2].astype(int)
    else:
        df_clean["HOUR"] = np.random.randint(0, 24, len(df_clean))

    df_clean["IS_WEEKEND"] = (df_clean["DAY_OF_WEEK"] >= 5).astype(int)
    df_clean["IS_RUSH_HOUR"] = ((df_clean["HOUR"] >= 7) & (df_clean["HOUR"] <= 9)).astype(int)

    feature_cols = [
        "MONTH", "DAY_OF_WEEK", "HOUR", "IS_WEEKEND", "IS_RUSH_HOUR",
        "DEP_DELAY", "DISTANCE", "CRS_ELAPSED_TIME"
    ]
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    X = df_clean[feature_cols].copy()
    y = df_clean["IS_DELAYED"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(model_key, params, experiment_name):
    """Train a model, log to MLflow, and return run details."""
    data_path = os.path.join(project_root, "2009.csv", "2009.csv")
    sample_frac = float(params.pop("sample_frac", MODEL_SAMPLE_FRAC.get(model_key, DEFAULT_SAMPLE_FRAC)))

    X_train, X_test, y_train, y_test = load_training_data(
        data_path=data_path,
        sample_frac=sample_frac,
        chunk_size=CHUNK_SIZE,
        random_state=RANDOM_STATE
    )

    scaler = None
    model = None
    log_params = {"sample_frac": sample_frac}

    if model_key == "random-forest":
        n_estimators = int(params.get("n_estimators", 100))
        max_depth = params.get("max_depth", None)
        max_depth = None if max_depth in [None, "", "null"] else int(max_depth)
        min_samples_split = int(params.get("min_samples_split", 2))

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        )
        log_params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else "None",
            "min_samples_split": min_samples_split,
            "class_weight": "balanced"
        })

    elif model_key == "svm":
        kernel = params.get("kernel", "rbf")
        gamma = params.get("gamma", "scale")
        c_value = float(params.get("C", 1.0))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(
            kernel=kernel,
            gamma=gamma,
            C=c_value,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        log_params.update({
            "kernel": kernel,
            "gamma": gamma,
            "C": c_value,
            "class_weight": "balanced"
        })

    elif model_key == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed in the current environment")

        n_estimators = int(params.get("n_estimators", 200))
        max_depth = int(params.get("max_depth", 6))
        learning_rate = float(params.get("learning_rate", 0.1))
        subsample = float(params.get("subsample", 0.8))

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        log_params.update({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample
        })

    elif model_key == "adaboost":
        n_estimators = int(params.get("n_estimators", 200))
        learning_rate = float(params.get("learning_rate", 0.5))

        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=RANDOM_STATE
        )
        log_params.update({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        })

    else:
        raise ValueError(f"Unsupported model: {model_key}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        y_pred_proba = None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else roc_auc_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
    mlflow.set_experiment(experiment_name)

    run_name = f"{MODEL_LABELS.get(model_key, model_key)}_run"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", input_example=X_test[:5])
        run_id = mlflow.active_run().info.run_id

    return {
        "run_id": run_id,
        "metrics": metrics,
        "params": log_params,
        "experiment_name": experiment_name
    }

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Get all MLflow experiments"""
    try:
        experiments = mlflow.search_experiments()
        experiments_data = []
        
        for exp in experiments:
            exp_data = {
                'id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'tags': exp.tags or {}
            }
            experiments_data.append(exp_data)
        
        return jsonify({
            'success': True,
            'experiments': experiments_data,
            'total': len(experiments_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/runs/<experiment_id>', methods=['GET'])
def get_runs(experiment_id):
    """Get all runs for a specific experiment"""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        runs_data = []

        for _, run in runs.iterrows():
            run_data = {
                'run_id': str(run['run_id']),
                'experiment_id': str(run['experiment_id']),
                'status': str(run['status']),
                'metrics': {},
                'params': {}
            }

            for col in runs.columns:
                if col.startswith('metrics.'):
                    metric_name = col.replace('metrics.', '')
                    try:
                        run_data['metrics'][metric_name] = float(run[col]) if run[col] is not None else 0
                    except:
                        pass
                elif col.startswith('params.'):
                    param_name = col.replace('params.', '')
                    run_data['params'][param_name] = str(run[col])

            runs_data.append(run_data)

        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'runs': runs_data,
            'total': len(runs_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/run/<run_id>', methods=['GET'])
def get_run_details(run_id):
    """Get detailed information for a specific run"""
    try:
        run = mlflow.get_run(run_id)
        
        run_data = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'duration': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
            'metrics': run.data.metrics,
            'params': run.data.params,
            'tags': run.data.tags,
            'artifacts': []
        }
        
        return jsonify({
            'success': True,
            'run': run_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary statistics of all experiments and runs"""
    try:
        experiments = mlflow.search_experiments()
        
        summary_data = {
            'total_experiments': len(experiments),
            'total_runs': 0,
            'completed_runs': 0,
            'failed_runs': 0,
            'experiments_list': []
        }
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            exp_summary = {
                'name': exp.name,
                'id': exp.experiment_id,
                'run_count': len(runs),
                'completed_runs': len(runs[runs['status'] == 'FINISHED']),
                'failed_runs': len(runs[runs['status'] == 'FAILED'])
            }
            summary_data['experiments_list'].append(exp_summary)
            summary_data['total_runs'] += len(runs)
            summary_data['completed_runs'] += len(runs[runs['status'] == 'FINISHED'])
            summary_data['failed_runs'] += len(runs[runs['status'] == 'FAILED'])
        
        return jsonify({
            'success': True,
            'summary': summary_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mlflow_tracking_uri': mlflow.get_tracking_uri(),
        'mlflow_db_exists': os.path.exists(mlflow_db_path)
    })

def _start_training(model_key, data):
    if model_key not in TRAINING_STATE:
        return jsonify({'success': False, 'error': 'Unsupported model'}), 404

    if TRAINING_STATE[model_key]['status'] == 'training':
        return jsonify({'success': False, 'error': 'Training already in progress'}), 409

    experiment_name = data.get('experiment_name', DEFAULT_EXPERIMENT)
    params = {k: v for k, v in data.items() if k != 'experiment_name'}

    TRAINING_STATE[model_key].update({
        'status': 'training',
        'run_id': None,
        'metrics': None,
        'params': None,
        'experiment_id': None,
        'error': None
    })

    def train_in_background():
        try:
            result = train_model(model_key, params, experiment_name)
            experiment = mlflow.get_experiment_by_name(result['experiment_name'])

            TRAINING_STATE[model_key].update({
                'status': 'complete',
                'run_id': result['run_id'],
                'metrics': result['metrics'],
                'params': result['params'],
                'experiment_id': experiment.experiment_id if experiment else None,
                'error': None
            })
        except Exception as exc:
            TRAINING_STATE[model_key].update({
                'status': 'error',
                'error': str(exc)
            })

    thread = threading.Thread(target=train_in_background, daemon=True)
    thread.start()

    return jsonify({'success': True, 'message': 'Training started', 'status': 'training'}), 202


@app.route('/api/train/<model_key>', methods=['POST'])
def train_model_endpoint(model_key):
    """Start training for a supported model"""
    try:
        data = request.json or {}
        return _start_training(model_key, data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train/<model_key>/status', methods=['GET'])
def get_training_status(model_key):
    """Get training status for a supported model"""
    try:
        if model_key not in TRAINING_STATE:
            return jsonify({'status': 'error', 'error': 'Unsupported model'}), 404

        state = TRAINING_STATE[model_key]

        if state['status'] == 'training':
            return jsonify({'status': 'training', 'message': 'Training in progress...'})

        if state['status'] == 'complete':
            return jsonify({
                'status': 'complete',
                'run_id': state['run_id'],
                'metrics': state['metrics'],
                'params': state['params'],
                'experiment_id': state['experiment_id']
            })

        if state['status'] == 'error':
            return jsonify({'status': 'error', 'message': state['error'] or 'Training failed'}), 500

        return jsonify({'status': 'idle', 'message': 'No training run yet'})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/runs/<experiment_id>/explore', methods=['GET'])
def explore_runs(experiment_id):
    """Get runs sorted by a metric for the Explore view."""
    try:
        from mlflow.tracking import MlflowClient as _Client
        metric = request.args.get('metric', 'accuracy')
        client = _Client(tracking_uri=f"sqlite:///{mlflow_db_path}")
        runs = client.search_runs(
            [experiment_id],
            order_by=[f'metrics.{metric} DESC'],
            max_results=50
        )
        result = []
        for r in runs:
            result.append({
                'run_id':    r.info.run_id,
                'run_name':  r.info.run_name or r.info.run_id[:8],
                'status':    r.info.status,
                'start_time': r.info.start_time,
                'metrics':   r.data.metrics,
                'params':    r.data.params,
            })
        return jsonify({'success': True, 'runs': result, 'sorted_by': metric})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/run/<run_id>/confusion_matrix', methods=['GET'])
def get_confusion_matrix(run_id):
    """Return the confusion matrix PNG as a base64 string."""
    try:
        import base64
        from mlflow.tracking import MlflowClient as _Client
        client = _Client(tracking_uri=f"sqlite:///{mlflow_db_path}")

        def _find(rid, path=''):
            for a in client.list_artifacts(rid, path):
                if a.is_dir:
                    found = _find(rid, a.path)
                    if found:
                        return found
                elif 'confusion_matrix' in a.path.lower() or (a.path.lower().split('/')[-1].startswith('cm_') and a.path.lower().endswith('.png')):
                    return a.path
            return None

        cm_path = _find(run_id)
        if not cm_path:
            return jsonify({'success': False, 'error': 'No confusion matrix found'}), 404

        local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=cm_path)
        with open(local, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return jsonify({'success': True, 'image': b64, 'path': cm_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
