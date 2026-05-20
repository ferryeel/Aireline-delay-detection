"""
src/simulate_drift.py — Partie 6 : Détection du Data Drift
- Simulation d'un jeu de production drifté
- Rapport Evidently (HTML + métriques MLflow)
- KS-test par feature
- Déclenchement conditionnel du ré-entraînement
"""
import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

import mlflow
from mlflow.tracking import MlflowClient

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric
    EVIDENTLY_OK = True
except Exception:
    EVIDENTLY_OK = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, '2009.csv', '2009.csv')
MLFLOW_URI   = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT   = 'Flight_Delay_Monitoring'

SEUIL_DRIFT  = 0.30   # 30% colonnes driftées → ré-entraînement
SEUIL_WARN   = 0.15   # 15% → alerte

FEATURES     = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND',
                'IS_RUSH_HOUR', 'DEP_DELAY', 'DISTANCE']


# ── 6.1 Chargement et prétraitement ───────────────────────────────────────────
def load_data():
    print('[1/5] Chargement des données...')
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200_000, low_memory=False):
        chunks.append(chunk.sample(frac=1_000_000/7_000_000, random_state=42))
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

    features = [c for c in FEATURES if c in df.columns]
    return df[features]


# ── 6.2 Simulation du drift ────────────────────────────────────────────────────
def simulate_drift(X_ref):
    """Simule un jeu de données de production drifté."""
    print('[2/5] Simulation du drift...')
    X_prod = X_ref.copy()
    num_cols = X_prod.select_dtypes(include=np.number).columns.tolist()

    # Drift fort sur DEP_DELAY (décalage +20 min = conditions hivernales sévères)
    if 'DEP_DELAY' in num_cols:
        X_prod['DEP_DELAY'] = X_prod['DEP_DELAY'] * 1.8 + np.random.normal(20, 5, len(X_prod))

    # Drift modéré sur HOUR (vols de nuit plus fréquents)
    if 'HOUR' in num_cols:
        X_prod['HOUR'] = (X_prod['HOUR'] + np.random.normal(3, 1, len(X_prod))).clip(0, 23).astype(int)

    for col in ['DEP_DELAY', 'HOUR']:
        if col in num_cols:
            ref_mean = X_ref[col].mean()
            prod_mean = X_prod[col].mean()
            print(f'    {col:15s} | Ref mean={ref_mean:.2f} | Prod mean={prod_mean:.2f} | Delta={prod_mean-ref_mean:+.2f}')

    return X_prod


def _html_fallback(X_ref, X_prod, ks_results_path):
    """Génère un rapport HTML basique quand Evidently n'est pas disponible."""
    rows = ''
    try:
        df_r = pd.read_csv(ks_results_path)
        for _, row in df_r.iterrows():
            color = '#ffe0e0' if row['drifted'] else '#e0ffe0'
            flag = 'DRIFT' if row['drifted'] else 'OK'
            rows += (f'<tr style="background:{color}">'
                     f'<td>{row["feature"]}</td><td>{row["ks_stat"]:.4f}</td>'
                     f'<td>{row["p_value"]:.6f}</td><td><b>{flag}</b></td></tr>')
    except Exception:
        rows = '<tr><td colspan="4">Données non disponibles</td></tr>'

    html = f"""<!DOCTYPE html><html lang="fr">
<head><meta charset="utf-8"><title>Drift Report</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:60%}}
th,td{{padding:8px 12px;border:1px solid #ccc;text-align:left}}th{{background:#333;color:#fff}}</style>
</head><body>
<h1>Rapport Data Drift — KS-test</h1>
<p><em>Evidently indisponible — rapport généré par KS-test (scipy.stats.ks_2samp)</em></p>
<h2>Résultats par feature</h2>
<table><tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>Statut</th></tr>
{rows}
</table>
<p>Drift détecté si p-value &lt; 0.05</p>
</body></html>"""
    return html


# ── 6.3 Evidently ─────────────────────────────────────────────────────────────
def run_evidently(X_ref, X_prod, run):
    drift_share, n_drifted, n_total = None, None, None
    if not EVIDENTLY_OK:
        return drift_share, n_drifted, n_total

    print('[3/5] Rapport Evidently...')

    # Rapport HTML visuel complet
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=X_ref, current_data=X_prod)
    html_path = os.path.join(PROJECT_ROOT, 'drift_report.html')
    report.save_html(html_path)
    mlflow.log_artifact(html_path, artifact_path='drift_artifacts')
    print(f'    Rapport HTML sauvegardé : drift_report.html')

    # Métriques numériques
    score_report = Report(metrics=[DatasetDriftMetric()])
    score_report.run(reference_data=X_ref, current_data=X_prod)
    result = score_report.as_dict()
    res    = result['metrics'][0]['result']

    drift_share = res['drift_share']
    n_drifted   = res['number_of_drifted_columns']
    n_total     = res['number_of_columns']

    mlflow.log_metric('evidently_drift_share',    drift_share)
    mlflow.log_metric('evidently_drifted_cols',   n_drifted)
    mlflow.log_metric('evidently_total_cols',     n_total)
    mlflow.log_metric('evidently_dataset_drifted', int(res['dataset_drift']))
    print(f'    Drift share : {drift_share:.2%} | Colonnes driftées : {n_drifted}/{n_total}')

    return drift_share, n_drifted, n_total


# ── 6.4 KS-test par feature ────────────────────────────────────────────────────
def run_ks_test(X_ref, X_prod):
    print('[4/5] KS-test par feature...')
    results = []
    for col in X_ref.select_dtypes(include='number').columns:
        stat, pvalue = stats.ks_2samp(X_ref[col].dropna(), X_prod[col].dropna())
        drifted = pvalue < 0.05
        results.append({'feature': col, 'ks_stat': round(stat, 4),
                        'p_value': round(pvalue, 6), 'drifted': drifted})
        mlflow.log_metric(f'ks_pvalue_{col}', pvalue)
        mlflow.log_metric(f'ks_stat_{col}',   stat)
        flag = ' << DRIFT' if drifted else ''
        print(f'    {col:15s} | stat={stat:.4f} | p-value={pvalue:.6f}{flag}')

    df_drift = pd.DataFrame(results)
    csv_path = os.path.join(PROJECT_ROOT, 'ks_drift_results.csv')
    df_drift.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path='drift_artifacts')

    ks_drift_share = df_drift['drifted'].mean()
    mlflow.log_metric('ks_drift_share', ks_drift_share)
    print(f'    KS drift share : {ks_drift_share:.2%} ({df_drift["drifted"].sum()}/{len(df_drift)} features)')
    return ks_drift_share, df_drift


# ── 6.5 Déclenchement automatique ─────────────────────────────────────────────
def maybe_retrain(drift_share, source='KS-test'):
    print(f'\n[5/5] Logique de ré-entraînement ({source}, seuil={SEUIL_DRIFT:.0%})...')
    if drift_share > SEUIL_DRIFT:
        print(f'  [CRITIQUE] Drift {drift_share:.2%} > {SEUIL_DRIFT:.0%} — ré-entraînement déclenché')
        mlflow.log_metric('retrain_triggered', 1)
        retrain_script = os.path.join(PROJECT_ROOT, 'src', 'train.py')
        venv_python    = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')
        python_exe     = venv_python if os.path.exists(venv_python) else sys.executable
        subprocess.run([python_exe, retrain_script, '--retrain'], check=True)
    elif drift_share > SEUIL_WARN:
        print(f'  [WARN] Drift {drift_share:.2%} > {SEUIL_WARN:.0%} — surveillance renforcée (pas de ré-entraînement)')
        mlflow.log_metric('retrain_triggered', 0)
    else:
        print(f'  [OK] Drift {drift_share:.2%} ≤ {SEUIL_WARN:.0%} — modèle stable')
        mlflow.log_metric('retrain_triggered', 0)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('='*65)
    print('  DÉTECTION DATA DRIFT — Pipeline MLOps Tâche 5')
    print('='*65)

    X_all  = load_data()
    split  = int(len(X_all) * 0.8)
    X_ref  = X_all.iloc[:split].reset_index(drop=True)
    X_prod = simulate_drift(X_all.iloc[split:].reset_index(drop=True))

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name='drift_check_v1') as run:
        mlflow.log_params({
            'ref_size':        len(X_ref),
            'prod_size':       len(X_prod),
            'seuil_drift':     SEUIL_DRIFT,
            'seuil_warn':      SEUIL_WARN,
            'drift_simulation': 'DEP_DELAY x1.8+20, HOUR+3',
        })

        if not EVIDENTLY_OK:
            print('[3/5] Evidently indisponible (conflit pydantic) — rapport HTML genere par KS-test.')

        # Evidently
        ev_drift_share, ev_drifted, ev_total = run_evidently(X_ref, X_prod, run)

        # KS-test
        csv_path = os.path.join(PROJECT_ROOT, 'ks_drift_results.csv')
        ks_drift_share, df_drift = run_ks_test(X_ref, X_prod)

        # Générer rapport HTML fallback si Evidently absent
        if not EVIDENTLY_OK:
            html_path = os.path.join(PROJECT_ROOT, 'drift_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(_html_fallback(X_ref, X_prod, csv_path))
            mlflow.log_artifact(html_path, artifact_path='drift_artifacts')
            print(f'    Rapport HTML fallback sauvegarde : drift_report.html')

        # Utiliser KS comme référence si Evidently absent
        drift_share = ev_drift_share if ev_drift_share is not None else ks_drift_share
        source      = 'Evidently' if ev_drift_share is not None else 'KS-test'

        # Déclenchement conditionnel
        maybe_retrain(drift_share, source=source)

    # Résumé pipeline
    print('\n' + '='*65)
    print('  PIPELINE MLOPS COMPLET — Vue d\'ensemble')
    print('='*65)
    print(
        "  Donnees brutes\n"
        "      |\n"
        "  [Pretraitement]              -> features: MONTH, HOUR, DEP_DELAY...\n"
        "      |\n"
        "  [Entrainement + MLflow]      -> params, metriques, artefacts\n"
        "      |\n"
        "  [MLflow Model Registry]      -> Staging -> Production\n"
        "      |\n"
        "  [Serving API REST]           -> mlflow models serve / FastAPI\n"
        "      |\n"
        "  [Evidently + KS-test]        -> drift_share, rapport HTML\n"
        "      |\n"
        "  drift > 30% ? -- OUI -> re-entrainement (boucle fermee)\n"
        "               -- NON -> surveillance continue\n"
    )
    print(f'  [OK] Run MLflow : {run.info.run_id}')
    print(f'  Rapport drift   : drift_report.html')
    print(f'  KS-test CSV     : ks_drift_results.csv')


if __name__ == '__main__':
    main()
