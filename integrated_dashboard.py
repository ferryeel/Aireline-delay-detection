"""
Integrated Dashboard Server - MLflow API + RF Training + Web UI
"""
import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import mlflow
import threading

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from rf_trainer import get_trainer

app = Flask(__name__)
CORS(app)

# Configure MLflow
mlflow_db_path = os.path.join(str(project_root), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

# ============ MLflow API Endpoints ============

@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Get all MLflow experiments"""
    try:
        experiments = mlflow.search_experiments()
        experiments_data = [{
            'id': exp.experiment_id,
            'name': exp.name,
            'artifact_location': exp.artifact_location,
            'lifecycle_stage': exp.lifecycle_stage,
            'tags': exp.tags or {}
        } for exp in experiments]

        return jsonify({'success': True, 'experiments': experiments_data, 'total': len(experiments_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/runs/<experiment_id>', methods=['GET'])
def get_runs(experiment_id):
    """Get all runs for a specific experiment"""
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        runs_data = []

        for _, run in runs.iterrows():
            run_data = {
                'run_id': str(run['run_id']),
                'experiment_id': str(run['experiment_id']),
                'status': str(run['status']),
                'metrics': {},
                'params': {}
            }

            for col in runs.columns:
                if col.startswith('metrics.'):
                    metric_name = col.replace('metrics.', '')
                    try:
                        run_data['metrics'][metric_name] = float(run[col]) if run[col] is not None else 0
                    except:
                        pass
                elif col.startswith('params.'):
                    param_name = col.replace('params.', '')
                    run_data['params'][param_name] = str(run[col])

            runs_data.append(run_data)

        return jsonify({'success': True, 'experiment_id': experiment_id, 'runs': runs_data, 'total': len(runs_data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary of all experiments"""
    try:
        experiments = mlflow.search_experiments()
        summary_data = {
            'total_experiments': len(experiments),
            'total_runs': 0,
            'completed_runs': 0,
            'failed_runs': 0,
            'experiments_list': []
        }

        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            exp_summary = {
                'name': exp.name,
                'id': exp.experiment_id,
                'run_count': len(runs),
                'completed_runs': len(runs[runs['status'] == 'FINISHED']),
                'failed_runs': len(runs[runs['status'] == 'FAILED'])
            }
            summary_data['experiments_list'].append(exp_summary)
            summary_data['total_runs'] += len(runs)
            summary_data['completed_runs'] += len(runs[runs['status'] == 'FINISHED'])
            summary_data['failed_runs'] += len(runs[runs['status'] == 'FAILED'])

        return jsonify({'success': True, 'summary': summary_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mlflow_tracking_uri': mlflow.get_tracking_uri(),
        'mlflow_db_exists': os.path.exists(mlflow_db_path)
    })

# ============ Training Endpoints ============

@app.route('/api/train/random-forest', methods=['POST'])
def train_random_forest():
    """Start Random Forest training"""
    try:
        data = request.json or {}
        n_estimators = int(data.get('n_estimators', 100))
        max_depth = int(data.get('max_depth', 10))
        min_samples_split = int(data.get('min_samples_split', 2))

        trainer = get_trainer()

        if trainer.status == 'training':
            return jsonify({'success': False, 'error': 'Training already in progress'}), 409

        def train_in_background():
            trainer.train(n_estimators, max_depth, min_samples_split)

        thread = threading.Thread(target=train_in_background, daemon=True)
        thread.start()

        return jsonify({'success': True, 'message': 'Training started', 'status': 'training'}), 202

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train/random-forest/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    try:
        trainer = get_trainer()

        if trainer.status == 'training':
            return jsonify({'status': 'training', 'message': 'Training in progress...'})

        experiment = mlflow.get_experiment_by_name('Flight_Delay_RF')
        if not experiment:
            return jsonify({'status': 'idle', 'message': 'No training run yet'})

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) == 0:
            return jsonify({'status': 'idle', 'message': 'No training run yet'})

        latest_run = runs.iloc[0]
        run_detail = mlflow.get_run(latest_run['run_id'])

        return jsonify({
            'status': 'complete' if trainer.status == 'complete' else 'idle',
            'run_id': latest_run['run_id'],
            'metrics': dict(run_detail.data.metrics),
            'params': dict(run_detail.data.params),
            'experiment_id': experiment.experiment_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*70)
    print("AeroPredict - Integrated Dashboard with RF Training")
    print("="*70)
    print(f"\n[*] Starting API on http://localhost:5000")
    print(f"[*] Dashboard on http://localhost:3000/index-mlflow.html")
    print(f"\nMLflow Database: {mlflow_db_path}")
    print("\nPress CTRL+C to stop\n")

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
