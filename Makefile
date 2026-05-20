# Makefile — Pipeline MLOps Local (Tâche 5)
# Usage : make <cible>   (ex: make pipeline)

PYTHON = .venv/Scripts/python.exe
MODEL  = flight_delay_rf_production
PORT   = 1234

.PHONY: setup train compare register serve test drift pipeline clean

## ── Environnement ─────────────────────────────────────────────────────────────
setup:
	$(PYTHON) -m pip install -r requirements.txt
	@echo "[OK] Dépendances installées."

## ── Entraînement principal (Partie 1) ────────────────────────────────────────
train:
	$(PYTHON) src/train.py
	@echo "[OK] Entraînement terminé et loggué dans MLflow."

## ── Comparaison 4 algorithmes (Partie 2) ─────────────────────────────────────
compare:
	$(PYTHON) src/compare_models.py
	@echo "[OK] Comparaison terminée. Lancez : make ui"

## ── Model Registry Staging → Production (Partie 3) ──────────────────────────
register:
	$(PYTHON) src/register_best_model.py
	@echo "[OK] Modèle enregistré dans le Registry MLflow."

## ── Serving REST (Partie 4) ──────────────────────────────────────────────────
serve:
	@echo "[*] Démarrage du serveur MLflow sur le port $(PORT)..."
	@echo "    Arrêtez avec Ctrl+C"
	.venv/Scripts/mlflow models serve \
		-m "models:/$(MODEL)/Production" \
		--port $(PORT) \
		--no-conda

## ── Test de l'API (Partie 4) ─────────────────────────────────────────────────
test:
	$(PYTHON) src/test_api.py

## ── Détection drift (Partie 6) ───────────────────────────────────────────────
drift:
	$(PYTHON) src/simulate_drift.py
	@echo "[OK] Détection drift terminée. Voir drift_report.html"

## ── Interface MLflow UI ──────────────────────────────────────────────────────
ui:
	@echo "[*] MLflow UI → http://localhost:5000"
	.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

## ── Pipeline complet ─────────────────────────────────────────────────────────
pipeline: train compare register drift
	@echo ""
	@echo "================================================================="
	@echo "  PIPELINE MLOPS COMPLET EXÉCUTÉ AVEC SUCCÈS"
	@echo "  Prochaine étape : make serve  puis  make test"
	@echo "================================================================="

## ── Nettoyage ─────────────────────────────────────────────────────────────────
clean:
	rm -f confusion_matrix.png classification_report.txt
	rm -f drift_report.html ks_drift_results.csv
	@echo "[OK] Fichiers temporaires supprimés."
