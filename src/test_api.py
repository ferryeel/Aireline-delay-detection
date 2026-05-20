"""
src/test_api.py — Partie 4 : Test de l'endpoint REST MLflow
Lance d'abord le serveur : mlflow models serve -m "models:/flight_delay_rf_production/Production" --port 1234 --no-conda
"""
import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import requests
except ImportError:
    print('[INSTALL] pip install requests')
    sys.exit(1)

API_URL = 'http://localhost:1234/invocations'

# Exemples de vols à prédire
# Features: MONTH, DAY_OF_WEEK, HOUR, IS_WEEKEND, IS_RUSH_HOUR, DEP_DELAY, DISTANCE
TEST_CASES = [
    {
        'label':   'Vol avec fort retard départ (prédit retardé)',
        'data':    [7, 4, 15, 0, 0, 45.0, 800.0],
    },
    {
        'label':   'Vol à l\'heure en début de matinée (prédit à l\'heure)',
        'data':    [3, 1, 7, 0, 1, -2.0, 500.0],
    },
    {
        'label':   'Vol week-end décembre soir (zone ambiguë)',
        'data':    [12, 6, 18, 1, 0, 8.0, 1200.0],
    },
]

COLUMNS = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND',
           'IS_RUSH_HOUR', 'DEP_DELAY', 'DISTANCE']


def test_endpoint():
    print('='*65)
    print('  TEST API REST — MLflow models serve')
    print('='*65)
    print(f'  Endpoint : {API_URL}\n')

    # Vérifier que le serveur est actif
    try:
        resp = requests.get('http://localhost:1234/ping', timeout=3)
        print(f'  Serveur actif : HTTP {resp.status_code}\n')
    except requests.exceptions.ConnectionError:
        print('  [ERREUR] Le serveur MLflow n\'est pas démarré.')
        print('  Lancez d\'abord :')
        print('    mlflow models serve -m "models:/flight_delay_rf_production/Production" --port 1234 --no-conda')
        print('  Ou depuis le Makefile : make serve')
        sys.exit(1)

    # Test avec chaque cas
    for i, case in enumerate(TEST_CASES, 1):
        payload = {
            'dataframe_split': {
                'columns': COLUMNS,
                'data':    [case['data']]
            }
        }

        t0   = time.time()
        resp = requests.post(API_URL, json=payload,
                             headers={'Content-Type': 'application/json'})
        ms   = (time.time() - t0) * 1000

        print(f'  [Test {i}] {case["label"]}')
        print(f'    Features : {dict(zip(COLUMNS, case["data"]))}')

        if resp.status_code == 200:
            result = resp.json()
            preds  = result.get('predictions', result)
            label  = 'RETARDÉ' if preds[0] == 1 else 'À L\'HEURE'
            print(f'    Prédiction : {label} (classe {preds[0]})  — {ms:.1f}ms')
        else:
            print(f'    [ERREUR] HTTP {resp.status_code} : {resp.text[:200]}')
        print()

    # Test batch (2 vols simultanés)
    print('  [Test batch] 2 vols simultanés...')
    batch_payload = {
        'dataframe_split': {
            'columns': COLUMNS,
            'data':    [TEST_CASES[0]['data'], TEST_CASES[1]['data']]
        }
    }
    resp = requests.post(API_URL, json=batch_payload)
    if resp.status_code == 200:
        preds = resp.json().get('predictions', resp.json())
        labels = ['RETARDÉ' if p == 1 else 'À L\'HEURE' for p in preds]
        print(f'    Résultats batch : {labels}')
    print()
    print('  [OK] Tests API terminés.')


if __name__ == '__main__':
    test_endpoint()
