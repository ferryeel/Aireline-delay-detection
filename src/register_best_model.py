"""
src/register_best_model.py — Partie 3 : Model Registry
Enregistre le meilleur run, transite Staging → Production avec validation.
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_URI       = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"
EXPERIMENT       = 'Flight_Delay_MLOps'
MODEL_NAME       = 'flight_delay_rf_production'
SEUIL_PRODUCTION = 0.85


def get_best_run(client, experiment_name):
    exp  = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=['metrics.accuracy DESC'],
        max_results=1
    )
    if not runs:
        raise RuntimeError(f"Aucun run trouvé dans l'expérience '{experiment_name}'")
    return runs[0]


def register_and_promote():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    # ── Étape 1 : Trouver le meilleur run ─────────────────────────────────────
    print('[1/4] Recherche du meilleur run...')
    best = get_best_run(client, EXPERIMENT)
    acc  = best.data.metrics.get('accuracy', 0)
    f1   = best.data.metrics.get('f1_score', 0)
    auc  = best.data.metrics.get('roc_auc', 0)
    print(f'  Run         : {best.info.run_name}')
    print(f'  Run ID      : {best.info.run_id}')
    print(f'  Accuracy    : {acc:.4f}')
    print(f'  F1-score    : {f1:.4f}')
    print(f'  ROC-AUC     : {auc:.4f}')

    # ── Étape 2 : Enregistrer dans le Registry ────────────────────────────────
    print(f'\n[2/4] Enregistrement dans le Model Registry (nom: {MODEL_NAME})...')
    model_uri  = f'runs:/{best.info.run_id}/model'
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version    = registered.version
    print(f'  Version enregistrée : v{version}')

    # Ajouter description et tags
    client.update_registered_model(
        name=MODEL_NAME,
        description='Modèle RF de classification des retards de vols 2009 — pipeline MLOps Tâche 5'
    )
    client.set_model_version_tag(MODEL_NAME, version, 'validated_by', 'pipeline_mlops')
    client.set_model_version_tag(MODEL_NAME, version, 'dataset',      '2009_flights')
    client.set_model_version_tag(MODEL_NAME, version, 'accuracy',     f'{acc:.4f}')
    print('  Tags et description ajoutés.')

    # ── Étape 3 : Promouvoir en Staging ──────────────────────────────────────
    print(f'\n[3/4] Promotion en Staging...')
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version,
        stage='Staging', archive_existing_versions=False
    )
    print(f'  Modele v{version} -> Staging [OK]')

    # ── Étape 4 : Validation et promotion en Production ───────────────────────
    print(f'\n[4/4] Validation (seuil accuracy >= {SEUIL_PRODUCTION})...')
    if acc >= SEUIL_PRODUCTION:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version,
            stage='Production', archive_existing_versions=True
        )
        print(f'  Modele v{version} -> Production [OK]  (accuracy={acc:.4f} >= {SEUIL_PRODUCTION})')
    else:
        print(f'  Non promu : accuracy={acc:.4f} < seuil {SEUIL_PRODUCTION}')
        print(f'  Modèle reste en Staging pour révision manuelle.')

    # ── Résumé ─────────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  RÉSUMÉ MODEL REGISTRY')
    print('='*60)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f'  v{v.version:>3} | {v.current_stage:<12} | {v.run_id[:8]}...')

    # URI pour serving
    prod_uri = f'models:/{MODEL_NAME}/Production'
    print(f'\n  URI Production : {prod_uri}')
    print(f'  Serving       : mlflow models serve -m "{prod_uri}" --port 1234 --no-conda')
    return version


if __name__ == '__main__':
    version = register_and_promote()
    print(f'\n[OK] Modèle v{version} enregistré et promu.')
