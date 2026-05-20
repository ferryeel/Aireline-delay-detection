"""
Random Forest trainer module for MLflow integration
"""
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


class RFTrainer:
    def __init__(self, data_path='2009.csv/2009.csv', mlflow_db_path='mlflow.db'):
        self.data_path = data_path
        self.mlflow_db_path = mlflow_db_path
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        self.status = 'idle'
        self.current_run = None

    def load_data(self):
        """Load and preprocess flight delay data"""
        print("[RF] Loading data...")
        SAMPLE_FRAC = 1_000_000 / 7_000_000
        CHUNK_SIZE = 200_000

        chunks = []
        for chunk in pd.read_csv(self.data_path, chunksize=CHUNK_SIZE, low_memory=False):
            chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

        df_raw = pd.concat(chunks, ignore_index=True)

        df_raw['IS_DELAYED'] = (df_raw['ARR_DELAY'] >= 15).astype(int)
        df_clean = df_raw[(df_raw['CANCELLED'] == 0) & (df_raw['DIVERTED'] == 0)].copy()

        drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
                     'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
        df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        date_col = 'FL_DATE' if 'FL_DATE' in df_clean.columns else None
        if date_col:
            df_clean['MONTH'] = pd.to_datetime(df_clean[date_col], errors='coerce').dt.month
            df_clean['DAY_OF_WEEK'] = pd.to_datetime(df_clean[date_col], errors='coerce').dt.dayofweek
        else:
            df_clean['MONTH'] = np.random.randint(1, 13, len(df_clean))
            df_clean['DAY_OF_WEEK'] = np.random.randint(0, 7, len(df_clean))

        if 'CRS_DEP_TIME' in df_clean.columns:
            df_clean['HOUR'] = df_clean['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
        else:
            df_clean['HOUR'] = np.random.randint(0, 24, len(df_clean))

        df_clean['IS_WEEKEND'] = (df_clean['DAY_OF_WEEK'] >= 5).astype(int)
        df_clean['IS_RUSH_HOUR'] = ((df_clean['HOUR'] >= 7) & (df_clean['HOUR'] <= 9)).astype(int)

        feature_cols = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND', 'IS_RUSH_HOUR',
                        'DEP_DELAY', 'DISTANCE']
        feature_cols = [c for c in feature_cols if c in df_clean.columns]

        X = df_clean[feature_cols].copy()
        y = df_clean['IS_DELAYED'].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train(self, n_estimators=100, max_depth=10, min_samples_split=2):
        """Train Random Forest and log to MLflow"""
        try:
            self.status = 'training'
            print("[RF] Loading training data...")
            X_train, X_test, y_train, y_test = self.load_data()

            experiment_name = "Flight_Delay_RF"
            try:
                exp = mlflow.get_experiment_by_name(experiment_name)
                exp_id = exp.experiment_id
            except:
                exp_id = mlflow.create_experiment(experiment_name)

            mlflow.set_experiment(experiment_name)

            print("[RF] Starting training with MLflow tracking...")
            with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}"):
                model_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'
                }

                mlflow.log_params(model_params)

                rf_model = RandomForestClassifier(**model_params)
                rf_model.fit(X_train, y_train)

                y_pred = rf_model.predict(X_test)
                y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(rf_model, "model", input_example=X_test.iloc[:5])

                with open('rf_model.pkl', 'wb') as f:
                    pickle.dump(rf_model, f)

                self.current_run = mlflow.active_run().info.run_id
                self.status = 'complete'

                print(f"[RF] Training complete: {metrics}")
                return {
                    'success': True,
                    'metrics': metrics,
                    'run_id': self.current_run,
                    'model_path': 'rf_model.pkl'
                }

        except Exception as e:
            self.status = 'error'
            print(f"[RF] Training error: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global trainer instance
trainer = None


def get_trainer():
    global trainer
    if trainer is None:
        trainer = RFTrainer()
    return trainer
