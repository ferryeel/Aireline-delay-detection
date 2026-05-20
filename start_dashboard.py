#!/usr/bin/env python3
"""
Unified Dashboard Server - Combines MLflow API + Web Interface
Run this to start the dashboard on http://localhost:3000
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Determine project root
project_root = Path(__file__).parent.absolute()

# MLflow API path
mlflow_api = project_root / "mlflow_api.py"
dashboard_dir = project_root / "dashboard"

print("""
╔══════════════════════════════════════════════════════════════╗
║         AeroPredict - Unified ML Dashboard                  ║
║     Flight Delay Prediction with MLflow Integration         ║
╚══════════════════════════════════════════════════════════════╝
""")

# Check if files exist
if not mlflow_api.exists():
    print("❌ Error: mlflow_api.py not found")
    sys.exit(1)

if not dashboard_dir.exists():
    print("❌ Error: dashboard directory not found")
    sys.exit(1)

print(f"📁 Project root: {project_root}")
print(f"🎨 Dashboard: {dashboard_dir}")
print()

# Start MLflow API
print("🚀 Starting MLflow API on http://localhost:5000...")
mlflow_process = subprocess.Popen(
    [sys.executable, str(mlflow_api)],
    cwd=str(project_root)
)

# Wait for MLflow API to start
print("⏳ Waiting for MLflow API to initialize...")
time.sleep(3)

# Start simple HTTP server for dashboard
print("🚀 Starting Web Server on http://localhost:3000...")
os.chdir(str(dashboard_dir))

try:
    # For Python 3
    import http.server
    import socketserver
    
    PORT = 3000
    Handler = http.server.SimpleHTTPRequestHandler
    
    def start_server():
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"✅ Dashboard ready at http://localhost:{PORT}")
            print("\n" + "="*60)
            print("📊 Open the following URLs in your browser:")
            print(f"   - Dashboard: http://localhost:{PORT}/index-mlflow.html")
            print(f"   - MLflow API: http://localhost:5000/health")
            print("="*60 + "\n")
            
            # Open browser
            time.sleep(1)
            try:
                webbrowser.open(f"http://localhost:{PORT}/index-mlflow.html")
            except:
                print(f"⚠️  Manually open http://localhost:{PORT}/index-mlflow.html in your browser")
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping servers...")
                mlflow_process.terminate()
                sys.exit(0)
    
    start_server()
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    mlflow_process.terminate()
    sys.exit(1)
