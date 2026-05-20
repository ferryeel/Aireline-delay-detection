"""
Analyse Biais-Variance - Grid Search Random Forest
n_estimators : [10, 50, 100, 200] x max_depth : [None, 5, 10, 20]
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

print("=" * 70)
print("  ANALYSE BIAIS-VARIANCE - GRID SEARCH RANDOM FOREST")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT & PRE-TRAITEMENT
# ============================================================
print("\n[1/4] Chargement du jeu de donnees...")
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
else:
    df_clean['MONTH']       = np.random.randint(1, 13, len(df_clean))
    df_clean['DAY_OF_WEEK'] = np.random.randint(0, 7,  len(df_clean))

if 'CRS_DEP_TIME' in df_clean.columns:
    df_clean['HOUR'] = df_clean['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)
else:
    df_clean['HOUR'] = np.random.randint(0, 24, len(df_clean))

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

# ============================================================
# 2. GRID SEARCH + MLFLOW
# ============================================================
N_ESTIMATORS_LIST = [10, 50, 100, 200]
MAX_DEPTH_LIST    = [None, 5, 10, 20]
TOTAL_RUNS        = len(N_ESTIMATORS_LIST) * len(MAX_DEPTH_LIST)

print(f"\n[2/4] Grid search : {TOTAL_RUNS} combinaisons...")
print(f"      n_estimators : {N_ESTIMATORS_LIST}")
print(f"      max_depth    : {MAX_DEPTH_LIST}\n")

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('RF_BiasVariance_Analysis')

results = []
run_idx = 0

for n_est in N_ESTIMATORS_LIST:
    for max_d in MAX_DEPTH_LIST:
        run_idx += 1
        depth_label = str(max_d) if max_d is not None else 'None'
        print(f"  [{run_idx:02d}/{TOTAL_RUNS}] n_estimators={n_est:>3}  max_depth={depth_label:>4}  ...", end='', flush=True)

        with mlflow.start_run(run_name=f"RF_ne{n_est}_md{depth_label}"):
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc  = accuracy_score(y_test,  model.predict(X_test))
            bias      = 1.0 - train_acc          # Biais = erreur sur train
            variance  = train_acc - test_acc      # Variance = gap train-test

            mlflow.log_params({
                'n_estimators': n_est,
                'max_depth': depth_label
            })
            mlflow.log_metrics({
                'train_accuracy': train_acc,
                'test_accuracy':  test_acc,
                'bias':           bias,
                'variance':       variance
            })

            results.append({
                'n_estimators': n_est,
                'max_depth':    max_d,
                'depth_label':  depth_label,
                'train_acc':    train_acc,
                'test_acc':     test_acc,
                'bias':         bias,
                'variance':     variance
            })

        print(f"  Train={train_acc:.4f}  Test={test_acc:.4f}  Biais={bias:.4f}  Variance={variance:.4f}")

# ============================================================
# 3. TABLEAU RECAPITULATIF
# ============================================================
print("\n[3/4] Tableau recapitulatif...")
print("\n" + "=" * 75)
print("  TABLEAU BIAIS-VARIANCE")
print("=" * 75)
header = f"  {'n_est':>6} | {'max_depth':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Biais':>8} | {'Variance':>10}"
print(header)
print("  " + "-" * 71)

df_res = pd.DataFrame(results)

for _, row in df_res.iterrows():
    overfitting = " <-- OVERFIT" if row['variance'] > 0.05 else ""
    underfit    = " <-- UNDERFIT" if row['train_acc'] < 0.80 else ""
    flag = overfitting or underfit
    print(f"  {int(row['n_estimators']):>6} | {row['depth_label']:>10} | "
          f"{row['train_acc']:>10.4f} | {row['test_acc']:>10.4f} | "
          f"{row['bias']:>8.4f} | {row['variance']:>10.4f}{flag}")

# ============================================================
# 4. IDENTIFICATION DES CAS
# ============================================================
print("\n" + "=" * 75)
print("  IDENTIFICATION DES CAS EXTREMES")
print("=" * 75)

# Overfitting : variance (gap) maximale
overfit_row = df_res.loc[df_res['variance'].idxmax()]
# Underfitting : train_acc minimale
underfit_row = df_res.loc[df_res['train_acc'].idxmin()]
# Equilibre : meilleur test_acc avec variance < 0.02
balanced_candidates = df_res[df_res['variance'] < 0.02]
if len(balanced_candidates) > 0:
    balanced_row = balanced_candidates.loc[balanced_candidates['test_acc'].idxmax()]
else:
    balanced_row = df_res.loc[df_res['test_acc'].idxmax()]

print(f"""
  [OVERFITTING] n_estimators={int(overfit_row['n_estimators'])}  max_depth={overfit_row['depth_label']}
    Train Acc = {overfit_row['train_acc']:.4f}  |  Test Acc = {overfit_row['test_acc']:.4f}
    Variance  = {overfit_row['variance']:.4f}  (Train >> Test)

  [UNDERFITTING] n_estimators={int(underfit_row['n_estimators'])}  max_depth={underfit_row['depth_label']}
    Train Acc = {underfit_row['train_acc']:.4f}  |  Test Acc = {underfit_row['test_acc']:.4f}
    Biais     = {underfit_row['bias']:.4f}  (Train Acc faible)

  [EQUILIBRE] n_estimators={int(balanced_row['n_estimators'])}  max_depth={balanced_row['depth_label']}
    Train Acc = {balanced_row['train_acc']:.4f}  |  Test Acc = {balanced_row['test_acc']:.4f}
    Biais     = {balanced_row['bias']:.4f}  |  Variance = {balanced_row['variance']:.4f}
""")

# ============================================================
# 5. GRAPHIQUE : Train vs Test Acc pour n_estimators=100
# ============================================================
print("[4/4] Generation du graphique...")

subset_100 = df_res[df_res['n_estimators'] == 100].copy()
# Pour le graphe, on convertit max_depth=None en valeur numerique (arbres complets = depth tres grande)
depth_numeric = [30 if d is None else d for d in subset_100['max_depth']]
depth_labels  = [str(d) if d is not None else 'None\n(complet)' for d in subset_100['max_depth']]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(range(len(depth_numeric)), subset_100['train_acc'].values,
        'o-', color='#2563EB', linewidth=2.5, markersize=9,
        markerfacecolor='white', markeredgewidth=2.5, label='Train Accuracy')
ax.plot(range(len(depth_numeric)), subset_100['test_acc'].values,
        's-', color='#DC2626', linewidth=2.5, markersize=9,
        markerfacecolor='white', markeredgewidth=2.5, label='Test Accuracy')

# Zone de variance (gap)
ax.fill_between(range(len(depth_numeric)),
                subset_100['train_acc'].values,
                subset_100['test_acc'].values,
                alpha=0.12, color='#9333EA', label='Gap (Variance)')

# Annotations des valeurs
for i, (tr, te) in enumerate(zip(subset_100['train_acc'].values, subset_100['test_acc'].values)):
    ax.annotate(f'{tr:.4f}', (i, tr), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9, color='#2563EB', fontweight='bold')
    ax.annotate(f'{te:.4f}', (i, te), textcoords='offset points',
                xytext=(0, -16), ha='center', fontsize=9, color='#DC2626', fontweight='bold')

ax.set_xticks(range(len(depth_numeric)))
ax.set_xticklabels(depth_labels, fontsize=11)
ax.set_xlabel('max_depth', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Courbes Train/Test Accuracy en fonction de max_depth\n(n_estimators=100, random_state=42)',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([min(subset_100['test_acc'].min(), 0.70) - 0.02,
             max(subset_100['train_acc'].max(), 0.95) + 0.03])

# Annotations regions
y_min = ax.get_ylim()[0]
ax.axvspan(-0.4, 0.4, alpha=0.04, color='orange', label='Underfitting zone')
ax.text(0, y_min + 0.005, 'Underfitting\n(biais eleve)', ha='center',
        fontsize=8, color='#B45309', style='italic')
ax.text(3, y_min + 0.005, 'Overfitting\n(variance elevee)', ha='center',
        fontsize=8, color='#7C3AED', style='italic')

plt.tight_layout()
output_path = 'bias_variance_curve.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"      Graphique sauvegarde : {output_path}")

# ============================================================
# 6. EXPORT JSON pour le rapport MD
# ============================================================
print("\n" + "=" * 75)
print("  RESULTATS COMPLETS (copier dans le rapport)")
print("=" * 75)
for _, row in df_res.iterrows():
    print(f"  ne={int(row['n_estimators']):>3}  md={row['depth_label']:>4}  "
          f"train={row['train_acc']:.4f}  test={row['test_acc']:.4f}  "
          f"biais={row['bias']:.4f}  var={row['variance']:.4f}")

print(f"\n  OVERFIT  -> ne={int(overfit_row['n_estimators'])}  md={overfit_row['depth_label']}")
print(f"  UNDERFIT -> ne={int(underfit_row['n_estimators'])}  md={underfit_row['depth_label']}")
print(f"  EQUILIBRE-> ne={int(balanced_row['n_estimators'])}  md={balanced_row['depth_label']}")

print("\n" + "=" * 75)
print("[OK] Analyse biais-variance terminee - 16 runs logues dans MLflow.")
print("=" * 75)
