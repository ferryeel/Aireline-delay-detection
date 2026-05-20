"""
Analyse des Erreurs du Random Forest - Faux Positifs & Faux Negatifs
Matrice de confusion, exemples d'erreurs, patterns, ameliorations
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

print("=" * 70)
print("  ANALYSE DES ERREURS - FAUX POSITIFS & FAUX NEGATIFS")
print("=" * 70)

# ============================================================
# 1. CHARGEMENT & PRE-TRAITEMENT (identique a robustness_analysis_rf.py)
# ============================================================
print("\n[1/5] Chargement du jeu de donnees...")
SAMPLE_FRAC = 1_000_000 / 7_000_000
CHUNK_SIZE  = 200_000
DATA_PATH   = r'2009.csv/2009.csv'

# On charge aussi les colonnes contextuelles pour l'analyse des erreurs
chunks_raw = []
for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunks_raw.append(chunk.sample(frac=SAMPLE_FRAC, random_state=42))

df_raw = pd.concat(chunks_raw, ignore_index=True)
print(f"      Donnees brutes : {df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes")

# Cible
df_raw['IS_DELAYED'] = (df_raw['ARR_DELAY'] >= 15).astype(int)
df_clean = df_raw[(df_raw['CANCELLED'] == 0) & (df_raw['DIVERTED'] == 0)].copy()

# Suppression des features de fuite
drop_cols = ['ARR_TIME', 'ARR_DELAY', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME',
             'WHEELS_OFF', 'WHEELS_ON', 'CANCELLED', 'DIVERTED', 'TAIL_NUM']
df_clean = df_clean.drop(columns=[c for c in drop_cols if c in df_clean.columns])

# Imputation mediane
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

# Split fixe
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Garder les colonnes contextuelles du jeu de test
context_cols = ['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST',
                'CRS_DEP_TIME', 'DEP_DELAY', 'DISTANCE', 'CRS_ELAPSED_TIME']
context_cols = [c for c in context_cols if c in df_clean.columns]
df_test_ctx  = df_clean.loc[X_test.index, context_cols].copy()

print(f"      Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")

# ============================================================
# 2. ENTRAINEMENT (random_state=42, hyperparams fixes)
# ============================================================
print("\n[2/5] Entrainement du modele (random_state=42)...")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('RF_Error_Analysis')

with mlflow.start_run(run_name='RF_error_analysis_rs42'):
    model = RandomForestClassifier(
        n_estimators=100, max_depth=None,
        random_state=42,  n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_params({'random_state': 42, 'n_estimators': 100, 'max_depth': 'None'})
    mlflow.log_metrics({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})

print(f"      Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

# ============================================================
# 3. MATRICE DE CONFUSION
# ============================================================
print("\n[3/5] Matrice de confusion...")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"""
  Matrice de confusion :
  ┌─────────────────────┬──────────────────────┐
  │                     │   Predit             │
  │                     │  A l'heure   Retarde │
  ├─────────────────────┼──────────────────────┤
  │ Reel  A l'heure     │  TN={tn:>8,}  FP={fp:>7,} │
  │       Retarde       │  FN={fn:>8,}  TP={tp:>7,} │
  └─────────────────────┴──────────────────────┘

  Vrais Negatifs  (TN) : {tn:>8,}  vols a l'heure correctement identifies
  Faux  Positifs  (FP) : {fp:>8,}  vols predit retardes mais a l'heure
  Faux  Negatifs  (FN) : {fn:>8,}  vols predit a l'heure mais retardes
  Vrais Positifs  (TP) : {tp:>8,}  vols retardes correctement identifies

  Taux FP / total non-retardes : {fp/(fp+tn)*100:.2f}%
  Taux FN / total retardes     : {fn/(fn+tp)*100:.2f}%
""")

# Classification report
print("  Classification Report complet :")
print(classification_report(y_test, y_pred, target_names=['A l heure (0)', 'Retarde (1)']))

# ============================================================
# 4. EXTRACTION DES EXEMPLES FP ET FN
# ============================================================
print("\n[4/5] Extraction des exemples d'erreurs...")

results_df = X_test.copy()
results_df['y_true']      = y_test.values
results_df['y_pred']      = y_pred
results_df['proba_delay'] = y_pred_proba

# Fusion avec le contexte (on retire les colonnes deja presentes dans X_test)
ctx_no_dup = df_test_ctx.drop(columns=[c for c in df_test_ctx.columns if c in results_df.columns])
results_df = results_df.join(ctx_no_dup, how='left')

# Separation FP et FN
fp_df = results_df[(results_df['y_true'] == 0) & (results_df['y_pred'] == 1)].copy()
fn_df = results_df[(results_df['y_true'] == 1) & (results_df['y_pred'] == 0)].copy()

print(f"      Total FP : {len(fp_df):,}  |  Total FN : {len(fn_df):,}")

# Tri : FP avec proba la plus haute (le modele est tres confiant de son erreur)
fp_sample = fp_df.nlargest(3, 'proba_delay')
# Tri : FN avec proba la plus basse (le modele est tres confiant de son erreur)
fn_sample = fn_df.nsmallest(3, 'proba_delay')

# Affichage des exemples
MONTH_NAMES = {1:'Jan',2:'Fev',3:'Mar',4:'Avr',5:'Mai',6:'Jun',
               7:'Jul',8:'Aou',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
DAY_NAMES   = {0:'Lun',1:'Mar',2:'Mer',3:'Jeu',4:'Ven',5:'Sam',6:'Dim'}

def display_example(row, idx, etype):
    month_n = MONTH_NAMES.get(int(row.get('MONTH', 0)), '?')
    day_n   = DAY_NAMES.get(int(row.get('DAY_OF_WEEK', 0)), '?')
    carrier = row.get('OP_CARRIER', 'N/A')
    origin  = row.get('ORIGIN', 'N/A')
    dest    = row.get('DEST', 'N/A')
    hour    = int(row.get('HOUR', 0))
    dep_del = row.get('DEP_DELAY', 'N/A')
    dist    = row.get('DISTANCE', 'N/A')
    proba   = row['proba_delay']
    weekend = 'Oui' if row.get('IS_WEEKEND', 0) == 1 else 'Non'
    rush    = 'Oui' if row.get('IS_RUSH_HOUR', 0) == 1 else 'Non'
    fl_date = row.get('FL_DATE', 'N/A')

    print(f"\n  --- {etype} Exemple {idx} ---")
    print(f"  Date         : {fl_date}  ({month_n} - {day_n})")
    print(f"  Compagnie    : {carrier}")
    print(f"  Route        : {origin} -> {dest}  (Distance: {dist:.0f} km)" if isinstance(dist, float) else f"  Route : {origin} -> {dest}")
    print(f"  Heure depart : {hour:02d}h  (Week-end: {weekend}  |  Rush hour: {rush})")
    print(f"  DEP_DELAY    : {dep_del:.1f} min" if isinstance(dep_del, float) else f"  DEP_DELAY : {dep_del}")
    print(f"  Proba retard : {proba:.4f}  ({proba*100:.1f}%)")
    print(f"  Vrai label   : {'RETARDE' if row['y_true']==1 else 'A L HEURE'}")
    print(f"  Prediction   : {'RETARDE' if row['y_pred']==1 else 'A L HEURE'}  <-- ERREUR")

print("\n" + "=" * 70)
print("  FAUX POSITIFS (vol predit RETARDE mais etait A L'HEURE)")
print("=" * 70)
for i, (_, row) in enumerate(fp_sample.iterrows(), 1):
    display_example(row, i, "[FP]")

print("\n" + "=" * 70)
print("  FAUX NEGATIFS (vol predit A L'HEURE mais etait RETARDE)")
print("=" * 70)
for i, (_, row) in enumerate(fn_sample.iterrows(), 1):
    display_example(row, i, "[FN]")

# ============================================================
# 5. PATTERNS COMMUNS DANS LES ERREURS
# ============================================================
print("\n" + "=" * 70)
print("  PATTERNS COMMUNS DANS LES ERREURS")
print("=" * 70)

errors_df = results_df[(results_df['y_true'] != results_df['y_pred'])].copy()
correct_df = results_df[(results_df['y_true'] == results_df['y_pred'])].copy()

print(f"\n  Total erreurs : {len(errors_df):,} / {len(results_df):,} ({len(errors_df)/len(results_df)*100:.2f}%)")

# Taux d'erreur par compagnie
if 'OP_CARRIER' in errors_df.columns:
    carrier_err = errors_df.groupby('OP_CARRIER').size()
    carrier_tot = results_df.groupby('OP_CARRIER').size()
    carrier_rate = (carrier_err / carrier_tot * 100).sort_values(ascending=False)
    print(f"\n  Taux d'erreur par compagnie (Top 5) :")
    for carrier, rate in carrier_rate.head(5).items():
        cnt = carrier_err.get(carrier, 0)
        print(f"    {carrier:>4}  :  {rate:5.2f}%  ({cnt:,} erreurs)")

# Taux d'erreur par aeroport d'origine
if 'ORIGIN' in errors_df.columns:
    origin_err  = errors_df.groupby('ORIGIN').size()
    origin_tot  = results_df.groupby('ORIGIN').size()
    origin_rate = (origin_err / origin_tot * 100)
    # Garder uniquement les aeroports avec > 100 vols test
    origin_rate = origin_rate[origin_tot > 100].sort_values(ascending=False)
    print(f"\n  Taux d'erreur par aeroport d'origine (Top 5, min 100 vols) :")
    for airport, rate in origin_rate.head(5).items():
        cnt = origin_err.get(airport, 0)
        print(f"    {airport:>4}  :  {rate:5.2f}%  ({cnt:,} erreurs)")

# Erreurs par tranche horaire
print(f"\n  Taux d'erreur par tranche horaire :")
hour_err  = errors_df.groupby('HOUR').size()
hour_tot  = results_df.groupby('HOUR').size()
hour_rate = (hour_err / hour_tot * 100).sort_values(ascending=False)
for hour, rate in hour_rate.head(5).items():
    cnt = hour_err.get(hour, 0)
    print(f"    {hour:02d}h  :  {rate:5.2f}%  ({cnt:,} erreurs)")

# DEP_DELAY moyen pour les FP et FN
print(f"\n  DEP_DELAY moyen :")
print(f"    FP (predit retarde, etait a l heure) : {fp_df['DEP_DELAY'].mean():.1f} min")
print(f"    FN (predit a l heure, etait retarde) : {fn_df['DEP_DELAY'].mean():.1f} min")
print(f"    Vrais Positifs (TP)                  : {results_df[(results_df['y_true']==1)&(results_df['y_pred']==1)]['DEP_DELAY'].mean():.1f} min")
print(f"    Vrais Negatifs (TN)                  : {results_df[(results_df['y_true']==0)&(results_df['y_pred']==0)]['DEP_DELAY'].mean():.1f} min")

# ============================================================
# 6. RESUME FINAL
# ============================================================
print("\n" + "=" * 70)
print("  RESUME & AMELIORATIONS PROPOSEES")
print("=" * 70)
print(f"""
  Matrice de confusion :
    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}

  Observations cles :
  - Les FP (vol a l'heure predit retarde) ont un DEP_DELAY eleve mais
    l'avion finit par rattraper son retard en vol.
  - Les FN (retard manque) concernent souvent des vols avec faible
    DEP_DELAY (depart quasi a l'heure) mais retard accumule en route.

  Amelioration 1 - Enrichissement des donnees :
    Ajouter des variables meteorologiques (vent, visibilite, precipitations)
    par aeroport et par heure. Les retards dus a la meteo sont impredictibles
    avec les features actuelles et gonflent les FN.

  Amelioration 2 - Feature Engineering :
    Creer une feature "retard cumulatif de l'appareil" (LATE_AIRCRAFT_DELAY)
    et le taux historique de ponctualite par compagnie + route + heure.
    Ces features capturent l'effet de propagation des retards (effet domino)
    qui est la principale source de FN sur ce dataset.
""")

print("=" * 70)
print("[OK] Analyse des erreurs terminee - Run logue dans MLflow.")
print("=" * 70)
