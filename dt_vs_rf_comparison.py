"""
Comparaison Decision Tree vs Random Forest
Meilleurs hyperparametres identifies dans les experiences precedentes
RF: n_estimators=100, max_depth=10 (config equilibree)
DT: max_depth=10 (meme profondeur, meilleur compromis biais-variance)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import mlflow
import mlflow.sklearn
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)
warnings.filterwarnings('ignore')

print("=" * 70)
print("  COMPARAISON DECISION TREE vs RANDOM FOREST")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT & PRE-TRAITEMENT
# ============================================================
print("\n[1/5] Chargement du jeu de donnees...")
SAMPLE_FRAC = 1_000_000 / 7_000_000
CHUNK_SIZE  = 200_000
DATA_PATH   = r'2009.csv/2009.csv'

chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df_raw = pd.concat(chunks, ignore_index=True)
df_raw['IS_DELAYED'] = (df_raw['ARR_DELAY'] >= 15).astype(int)
df_clean = df_raw[(df_raw['CANCELLED'] == 0) & (df_raw['DIVERTED'] == 0)].copy()

drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
             'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

if 'FL_DATE' in df_clean.columns:
    df_clean['MONTH']       = pd.to_datetime(df_clean['FL_DATE'], errors='coerce').dt.month
    df_clean['DAY_OF_WEEK'] = pd.to_datetime(df_clean['FL_DATE'], errors='coerce').dt.dayofweek

if 'CRS_DEP_TIME' in df_clean.columns:
    df_clean['HOUR'] = df_clean['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)

df_clean['IS_WEEKEND']   = (df_clean['DAY_OF_WEEK'] >= 5).astype(int)
df_clean['IS_RUSH_HOUR'] = ((df_clean['HOUR'] >= 7) & (df_clean['HOUR'] <= 9)).astype(int)

feature_cols = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND', 'IS_RUSH_HOUR',
                'DEP_DELAY', 'DISTANCE']
feature_cols = [c for c in feature_cols if c in df_clean.columns]

X = df_clean[feature_cols].copy()
y = df_clean['IS_DELAYED'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
print(f"      Features : {feature_cols}")

# ============================================================
# 2. ENTRAINEMENT DES DEUX MODELES
# ============================================================
print("\n[2/5] Entrainement des modeles...")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('DT_vs_RF_Comparison')

results = {}

# --- Decision Tree ---
print("\n  [A] Decision Tree (max_depth=10)...")
t0 = time.time()
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_train_time = time.time() - t0
print(f"      Temps d'entrainement : {dt_train_time:.2f}s")

t0 = time.time()
dt_train_pred = dt_model.predict(X_train)
dt_train_inf  = (time.time() - t0) * 1000  # ms

t0 = time.time()
dt_test_pred  = dt_model.predict(X_test)
dt_test_inf   = (time.time() - t0) * 1000  # ms

dt_metrics = {
    'train_accuracy':  accuracy_score(y_train, dt_train_pred),
    'test_accuracy':   accuracy_score(y_test,  dt_test_pred),
    'train_precision': precision_score(y_train, dt_train_pred, zero_division=0),
    'test_precision':  precision_score(y_test,  dt_test_pred,  zero_division=0),
    'train_recall':    recall_score(y_train, dt_train_pred, zero_division=0),
    'test_recall':     recall_score(y_test,  dt_test_pred,  zero_division=0),
    'train_f1':        f1_score(y_train, dt_train_pred, zero_division=0),
    'test_f1':         f1_score(y_test,  dt_test_pred,  zero_division=0),
    'train_time_s':    dt_train_time,
    'test_inference_ms': dt_test_inf,
}

print(f"      Train Acc={dt_metrics['train_accuracy']:.4f}  Test Acc={dt_metrics['test_accuracy']:.4f}")
print(f"      Precision={dt_metrics['test_precision']:.4f}  Recall={dt_metrics['test_recall']:.4f}  F1={dt_metrics['test_f1']:.4f}")
print(f"      Inference test : {dt_test_inf:.1f}ms")

with mlflow.start_run(run_name='DecisionTree_md10'):
    mlflow.log_params({'model': 'DecisionTree', 'max_depth': 10, 'random_state': 42})
    mlflow.log_metrics(dt_metrics)
    mlflow.sklearn.log_model(dt_model, 'model')

results['Decision Tree'] = dt_metrics

# --- Random Forest ---
print("\n  [B] Random Forest (n_estimators=100, max_depth=10)...")
t0 = time.time()
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - t0
print(f"      Temps d'entrainement : {rf_train_time:.2f}s")

t0 = time.time()
rf_train_pred = rf_model.predict(X_train)
rf_train_inf  = (time.time() - t0) * 1000

t0 = time.time()
rf_test_pred  = rf_model.predict(X_test)
rf_test_inf   = (time.time() - t0) * 1000

rf_metrics = {
    'train_accuracy':  accuracy_score(y_train, rf_train_pred),
    'test_accuracy':   accuracy_score(y_test,  rf_test_pred),
    'train_precision': precision_score(y_train, rf_train_pred, zero_division=0),
    'test_precision':  precision_score(y_test,  rf_test_pred,  zero_division=0),
    'train_recall':    recall_score(y_train, rf_train_pred, zero_division=0),
    'test_recall':     recall_score(y_test,  rf_test_pred,  zero_division=0),
    'train_f1':        f1_score(y_train, rf_train_pred, zero_division=0),
    'test_f1':         f1_score(y_test,  rf_test_pred,  zero_division=0),
    'train_time_s':    rf_train_time,
    'test_inference_ms': rf_test_inf,
}

print(f"      Train Acc={rf_metrics['train_accuracy']:.4f}  Test Acc={rf_metrics['test_accuracy']:.4f}")
print(f"      Precision={rf_metrics['test_precision']:.4f}  Recall={rf_metrics['test_recall']:.4f}  F1={rf_metrics['test_f1']:.4f}")
print(f"      Inference test : {rf_test_inf:.1f}ms")

with mlflow.start_run(run_name='RandomForest_ne100_md10'):
    mlflow.log_params({'model': 'RandomForest', 'n_estimators': 100, 'max_depth': 10, 'random_state': 42})
    mlflow.log_metrics(rf_metrics)
    mlflow.sklearn.log_model(rf_model, 'model')

results['Random Forest'] = rf_metrics

# ============================================================
# 3. TABLEAU COMPARATIF
# ============================================================
print("\n[3/5] Tableau comparatif...")
print("\n" + "=" * 72)
print("  TABLEAU DE COMPARAISON DT vs RF")
print("=" * 72)

dt = results['Decision Tree']
rf = results['Random Forest']

def winner(a, b, higher_better=True):
    if higher_better:
        return "DT [+]" if a > b else "RF [+]"
    else:
        return "DT [+]" if a < b else "RF [+]"

rows = [
    ("Train Accuracy",    dt['train_accuracy'],   rf['train_accuracy'],   True),
    ("Test Accuracy",     dt['test_accuracy'],    rf['test_accuracy'],    True),
    ("Train Precision",   dt['train_precision'],  rf['train_precision'],  True),
    ("Test Precision",    dt['test_precision'],   rf['test_precision'],   True),
    ("Train Recall",      dt['train_recall'],     rf['train_recall'],     True),
    ("Test Recall",       dt['test_recall'],      rf['test_recall'],      True),
    ("Train F1-score",    dt['train_f1'],         rf['train_f1'],         True),
    ("Test F1-score",     dt['test_f1'],          rf['test_f1'],          True),
    ("Train Time (s)",    dt['train_time_s'],     rf['train_time_s'],     False),
    ("Inference (ms)",    dt['test_inference_ms'],rf['test_inference_ms'],False),
]

header = f"  {'Metrique':<20} | {'Decision Tree':>14} | {'Random Forest':>14} | {'Meilleur':>10}"
print(header)
print("  " + "-" * 68)
for name, dval, rval, hb in rows:
    fmt = ".4f" if name not in ["Train Time (s)", "Inference (ms)"] else ".2f"
    w = winner(dval, rval, hb)
    print(f"  {name:<20} | {dval:>14{fmt}} | {rval:>14{fmt}} | {w:>10}")

print("\n" + "=" * 72)
print(f"  Interpretabilite  | {'Elevee (1 arbre)':>14} | {'Faible (100 arbres)':>14} |")
print(f"  Memoire           | {'Faible':>14} | {'Elevee (~100x)':>14} |")
print("=" * 72)

# ============================================================
# 4. GRAPHIQUE RADAR
# ============================================================
print("\n[4/5] Generation du graphique radar + barres...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Subplot 1 : Radar chart (metriques test) ---
metrics_radar = ['Accuracy', 'Precision', 'Recall', 'F1-score']
dt_vals  = [dt['test_accuracy'], dt['test_precision'], dt['test_recall'], dt['test_f1']]
rf_vals  = [rf['test_accuracy'], rf['test_precision'], rf['test_recall'], rf['test_f1']]

angles   = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
dt_vals  += dt_vals[:1]
rf_vals  += rf_vals[:1]
angles   += angles[:1]

ax1 = plt.subplot(121, polar=True)
ax1.plot(angles, dt_vals, 'o-', linewidth=2.5, color='#F59E0B', label='Decision Tree')
ax1.fill(angles, dt_vals, alpha=0.2, color='#F59E0B')
ax1.plot(angles, rf_vals, 's-', linewidth=2.5, color='#2563EB', label='Random Forest')
ax1.fill(angles, rf_vals, alpha=0.2, color='#2563EB')
ax1.set_thetagrids(np.degrees(angles[:-1]), metrics_radar, fontsize=12)
ax1.set_ylim(0.60, 1.0)
ax1.set_yticks([0.65, 0.75, 0.85, 0.95])
ax1.set_yticklabels(['0.65', '0.75', '0.85', '0.95'], fontsize=8)
ax1.set_title('Metriques Test (Radar)\nDT vs RF', fontsize=13, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotations des valeurs sur le radar
for angle, dt_v, rf_v, label in zip(angles[:-1], dt_vals[:-1], rf_vals[:-1], metrics_radar):
    ax1.annotate(f'{dt_v:.3f}', xy=(angle, dt_v), fontsize=8, color='#B45309',
                 ha='center', va='center',
                 xytext=(np.cos(angle)*0.04, np.sin(angle)*0.04),
                 textcoords='offset points')

# --- Subplot 2 : Barres comparatives ---
ax2 = axes[1]
metric_names  = ['Test\nAccuracy', 'Test\nPrecision', 'Test\nRecall', 'Test\nF1-score']
dt_test_vals  = [dt['test_accuracy'], dt['test_precision'], dt['test_recall'], dt['test_f1']]
rf_test_vals  = [rf['test_accuracy'], rf['test_precision'], rf['test_recall'], rf['test_f1']]

x      = np.arange(len(metric_names))
width  = 0.35

bars_dt = ax2.bar(x - width/2, dt_test_vals, width, label='Decision Tree',
                  color='#F59E0B', alpha=0.85, edgecolor='white', linewidth=1.5)
bars_rf = ax2.bar(x + width/2, rf_test_vals, width, label='Random Forest',
                  color='#2563EB', alpha=0.85, edgecolor='white', linewidth=1.5)

for bar in bars_dt:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.4f}',
             ha='center', va='bottom', fontsize=9, color='#92400E', fontweight='bold')
for bar in bars_rf:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.003, f'{h:.4f}',
             ha='center', va='bottom', fontsize=9, color='#1E3A8A', fontweight='bold')

ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Metriques Test cote a cote\nDT vs RF', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metric_names, fontsize=11)
ax2.set_ylim(0.60, 1.02)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('Decision Tree vs Random Forest\n(max_depth=10 | Dataset : 2009 Flights)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()

output_path = 'dt_vs_rf_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"      Graphique sauvegarde : {output_path}")

# ============================================================
# 5. RESULTATS FINAUX
# ============================================================
print("\n[5/5] Synthese finale...")
print("\n" + "=" * 72)
print("  SYNTHESE COMPARATIVE")
print("=" * 72)

print(f"""
  Test Accuracy   : DT={dt['test_accuracy']:.4f}  RF={rf['test_accuracy']:.4f}  delta=+{rf['test_accuracy']-dt['test_accuracy']:.4f} (RF)
  Test Precision  : DT={dt['test_precision']:.4f}  RF={rf['test_precision']:.4f}  delta=+{rf['test_precision']-dt['test_precision']:.4f} (RF)
  Test Recall     : DT={dt['test_recall']:.4f}  RF={rf['test_recall']:.4f}  delta=+{rf['test_recall']-dt['test_recall']:.4f} (RF)
  Test F1-score   : DT={dt['test_f1']:.4f}  RF={rf['test_f1']:.4f}  delta=+{rf['test_f1']-dt['test_f1']:.4f} (RF)
  Train Time      : DT={dt['train_time_s']:.2f}s  RF={rf['train_time_s']:.2f}s  RF={rf['train_time_s']/dt['train_time_s']:.1f}x plus lent
  Inference       : DT={dt['test_inference_ms']:.1f}ms  RF={rf['test_inference_ms']:.1f}ms  RF={rf['test_inference_ms']/dt['test_inference_ms']:.1f}x plus lent
""")

print("=" * 72)
print("[OK] Comparaison terminee - 2 runs logues dans MLflow.")
print("=" * 72)
