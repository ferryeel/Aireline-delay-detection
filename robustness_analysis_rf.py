"""
Analyse de Robustesse - Random Forest Classifier
5 runs avec random_states [0, 42, 123, 256, 999]
Hyperparametres fixes : n_estimators=100, max_depth=None
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATES = [0, 42, 123, 256, 999]
N_ESTIMATORS  = 100
MAX_DEPTH     = None          # ← pas de limitation de profondeur
SAMPLE_FRAC   = 1_000_000 / 7_000_000
CHUNK_SIZE    = 200_000
DATA_PATH     = r'2009.csv/2009.csv'
EXPERIMENT    = "RF_Robustness_Analysis"

print("=" * 70)
print("  ANALYSE DE ROBUSTESSE – RANDOM FOREST (5 runs)")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT & PRÉ-TRAITEMENT
# ============================================================
print("\n[1/4] Chargement du jeu de données...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df_raw = pd.concat(chunks, ignore_index=True)
print(f"      Données brutes : {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")

# Cible
df_raw['IS_DELAYED'] = (df_raw['ARR_DELAY'] >= 15).astype(int)

# Filtrage des vols annulés / déroutés
df_clean = df_raw[(df_raw['CANCELLED'] == 0) & (df_raw['DIVERTED'] == 0)].copy()

# Suppression des features de fuite
drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
             'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

# Imputation médiane
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Features temporelles
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

# Split fixe (même test set pour tous les runs)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"      Features utilisées : {feature_cols}")
print(f"      Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
print(f"      Classe 1 (retardé) : {y.mean()*100:.1f}%")

# ============================================================
# 2. CONFIGURATION MLFLOW
# ============================================================
print("\n[2/4] Configuration MLflow...")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment(EXPERIMENT)
print(f"      Expérience : '{EXPERIMENT}'")

# ============================================================
# 3. ENTRAÎNEMENT DES 5 MODÈLES
# ============================================================
print("\n[3/4] Entraînement des 5 modèles...")
print("-" * 70)

results = []

for idx, rs in enumerate(RANDOM_STATES, 1):
    print(f"\n  Run {idx}/5 — random_state={rs}")

    with mlflow.start_run(run_name=f"RF_rs{rs}"):

        # Hyperparamètres identiques pour tous les runs
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=rs,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Métriques
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        # Log MLflow
        mlflow.log_params({
            'random_state': rs,
            'n_estimators': N_ESTIMATORS,
            'max_depth': str(MAX_DEPTH)
        })
        mlflow.log_metrics({
            'accuracy':  acc,
            'precision': prec,
            'recall':    rec,
            'f1_score':  f1
        })

        results.append({
            'random_state': rs,
            'accuracy':     acc,
            'precision':    prec,
            'recall':       rec,
            'f1_score':     f1
        })

        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(f"    F1-score  : {f1:.4f}")

# ============================================================
# 4. TABLEAU RÉCAPITULATIF & ANALYSE
# ============================================================
print("\n[4/4] Analyse de robustesse...")
print("\n" + "=" * 70)
print("TABLEAU RÉCAPITULATIF DES 5 RUNS")
print("=" * 70)

df_res = pd.DataFrame(results)

# Affichage tableau
header = f"{'random_state':>14} | {'accuracy':>10} | {'precision':>10} | {'recall':>10} | {'f1_score':>10}"
print(header)
print("-" * len(header))
for _, row in df_res.iterrows():
    print(f"{int(row['random_state']):>14} | {row['accuracy']:>10.4f} | "
          f"{row['precision']:>10.4f} | {row['recall']:>10.4f} | {row['f1_score']:>10.4f}")

print("-" * len(header))

# Statistiques agrégées
means  = df_res[['accuracy', 'precision', 'recall', 'f1_score']].mean()
stds   = df_res[['accuracy', 'precision', 'recall', 'f1_score']].std()

print(f"{'MEAN':>14} | {means['accuracy']:>10.4f} | {means['precision']:>10.4f} | "
      f"{means['recall']:>10.4f} | {means['f1_score']:>10.4f}")
print(f"{'STD DEV':>14} | {stds['accuracy']:>10.4f} | {stds['precision']:>10.4f} | "
      f"{stds['recall']:>10.4f} | {stds['f1_score']:>10.4f}")

# Verdict de robustesse (basé sur F1-score comme métrique globale)
f1_std = stds['f1_score']
acc_std = stds['accuracy']

print("\n" + "=" * 70)
print("VERDICT DE ROBUSTESSE")
print("=" * 70)
print(f"  Écart-type Accuracy  : {acc_std:.4f}")
print(f"  Écart-type F1-score  : {f1_std:.4f}")

if f1_std < 0.01:
    verdict = "ROBUSTE"
    marker  = "[OK]"
    interpretation = (
        "L'ecart-type du F1-score est inferieur a 0.01 sur 5 runs avec des random_states "
        "tres differents ([0, 42, 123, 256, 999]). Le modele produit des resultats coherents "
        "quelle que soit la graine aleatoire : le Random Forest est stable sur ce jeu de donnees."
    )
elif f1_std < 0.03:
    verdict = "ACCEPTABLE"
    marker  = "[~~]"
    interpretation = (
        "L'ecart-type du F1-score est compris entre 0.01 et 0.03. Les variations restent "
        "moderees, mais suggerent une legere sensibilite a la graine aleatoire. "
        "Verifier l'equilibre des classes et la qualite du feature engineering."
    )
else:
    verdict = "INSTABLE"
    marker  = "[!!]"
    interpretation = (
        "L'ecart-type du F1-score depasse 0.03, signe d'instabilite marquee. "
        "Sources possibles : desequilibre de classes, features bruitees (DEP_DELAY), "
        "variance elevee due a max_depth=None, ou taille d'echantillon insuffisante."
    )

print(f"\n  {marker}  Verdict : {verdict}")
print(f"\n  Interpretation :\n  {interpretation}")

print("\n" + "=" * 70)
print("[OK] Analyse terminée — tous les runs sont loggés dans MLflow.")
print("=" * 70)
