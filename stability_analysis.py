"""
Random Forest Stability Analysis - Training 5 Models with Different Random States
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RANDOM FOREST STABILITY ANALYSIS")
print("="*80)

# Configuration
RANDOM_STATES = [0, 7, 21, 42, 99]
N_ESTIMATORS = 100
MAX_DEPTH = 10
SAMPLE_FRAC = 1_000_000 / 7_000_000
CHUNK_SIZE = 200_000
DATA_PATH = r'2009.csv/2009.csv'

# Load and prepare data (same as training)
print("\n[*] Loading flight delay dataset...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df_raw = pd.concat(chunks, ignore_index=True)
print(f"[OK] Raw data loaded: {df_raw.shape}")

# Preprocessing
print("[*] Preprocessing data...")
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

# Train-test split (fixed for stability comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Features: {feature_cols}")
print(f"[OK] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Setup MLflow
print("\n[*] Setting up MLflow...")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
experiment_name = 'Flight_Delay_RF_Stability'

try:
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id
    print(f"[OK] Using existing experiment: {experiment_name}")
except:
    exp_id = mlflow.create_experiment(experiment_name)
    print(f"[OK] Created new experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)

# Train models with different random states
print("\n" + "="*80)
print("TRAINING 5 MODELS WITH DIFFERENT RANDOM STATES")
print("="*80)

results = []
models = {}

for idx, random_state in enumerate(RANDOM_STATES, 1):
    print(f"\n[{idx}/5] Training model with random_state={random_state}...")

    with mlflow.start_run(run_name=f"RF_stability_rs{random_state}"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Log to MLflow
        params = {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'random_state': random_state
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model', input_example=X_test.iloc[:5])

        # Store results
        result = {
            'random_state': random_state,
            **metrics
        }
        results.append(result)
        models[random_state] = model

        print(f"  [OK] Accuracy: {metrics['accuracy']:.4f}")
        print(f"       Precision: {metrics['precision']:.4f}")
        print(f"       Recall: {metrics['recall']:.4f}")
        print(f"       F1-Score: {metrics['f1_score']:.4f}")
        print(f"       ROC-AUC: {metrics['roc_auc']:.4f}")

# Analysis
print("\n" + "="*80)
print("STABILITY ANALYSIS")
print("="*80)

results_df = pd.DataFrame(results)

print(f"\n{'Metric':<15} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
print("-" * 55)

for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
    mean = results_df[metric].mean()
    std = results_df[metric].std()
    min_val = results_df[metric].min()
    max_val = results_df[metric].max()

    print(f"{metric:<15} {mean:<10.4f} {std:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")

# Detailed accuracy analysis
accuracy_values = results_df['accuracy'].values
accuracy_std = accuracy_values.std()
accuracy_mean = accuracy_values.mean()

print(f"\n{'='*55}")
print("ACCURACY STABILITY ASSESSMENT")
print(f"{'='*55}")
print(f"Accuracy Values: {[f'{acc:.4f}' for acc in accuracy_values]}")
print(f"Mean Accuracy: {accuracy_mean:.4f}")
print(f"Standard Deviation: {accuracy_std:.6f}")

if accuracy_std < 0.01:
    stability = "VERY STABLE"
    status = "[PASS]"
elif accuracy_std < 0.03:
    stability = "ACCEPTABLE"
    status = "[OK]"
else:
    stability = "UNSTABLE"
    status = "[WARN]"

print(f"Stability Assessment: {status} {stability}")

if accuracy_std < 0.01:
    print("Interpretation: Model produces consistent predictions across different random seeds.")
    print("                Excellent for production deployment.")
elif accuracy_std < 0.03:
    print("Interpretation: Model is reasonably stable with minor variations.")
    print("                Acceptable for most applications.")
else:
    print("Interpretation: Model shows significant variation across random seeds.")
    print("                Consider collecting more data or tuning hyperparameters.")

# Create visualization
print(f"\n[*] Creating visualization...")

fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
x = np.arange(len(RANDOM_STATES))
width = 0.17

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, metric in enumerate(metrics_to_plot):
    values = results_df[metric].values
    offset = (idx - 2) * width
    bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[idx], alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Random State', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Random Forest Model Stability - Metrics Across Different Random States\n(n_estimators=100, max_depth=10)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(RANDOM_STATES)
ax.set_ylim([0.7, 1.0])
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
output_file = 'stability_analysis.png'
print(f"[*] Saving figure to '{output_file}'...")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"[OK] Figure saved: {output_file}")

# Summary table
print("\n" + "="*80)
print("DETAILED RESULTS TABLE")
print("="*80)

summary_table = results_df.copy()
summary_table = summary_table.round(4)
print(summary_table.to_string(index=False))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total Models Trained: {len(RANDOM_STATES)}")
print(f"Hyperparameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
print(f"Experiment: {experiment_name}")
print(f"MLflow Runs: {len(RANDOM_STATES)}")
print(f"\nAccuracy Stability: {stability} (std: {accuracy_std:.6f})")
print(f"Average Accuracy: {accuracy_mean:.4f}")
print(f"Accuracy Range: [{accuracy_values.min():.4f}, {accuracy_values.max():.4f}]")

print("\n" + "="*80)
print("[OK] STABILITY ANALYSIS COMPLETE")
print("="*80)
