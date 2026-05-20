"""
Feature Importance Analysis for Random Forest Flight Delay Model
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Feature names used during training
FEATURE_NAMES = ['MONTH', 'DAY_OF_WEEK', 'HOUR', 'IS_WEEKEND', 'IS_RUSH_HOUR', 'DEP_DELAY', 'DISTANCE']

print("="*70)
print("FEATURE IMPORTANCE ANALYSIS - Random Forest")
print("="*70)

# Load the trained model
print("\n[*] Loading trained model from 'rf_model.pkl'...")
try:
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("[OK] Model loaded successfully")
except FileNotFoundError:
    print("[ERROR] rf_model.pkl not found. Please train the model first.")
    exit(1)

# Extract feature importances
print("\n[*] Extracting feature importances...")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Importance': importances
}).sort_values('Importance', ascending=True)

print("[OK] Feature importances extracted")

# Print top 3 features
print("\n" + "="*70)
print("TOP 3 MOST IMPORTANT FEATURES")
print("="*70)
top_3 = feature_importance_df.tail(3).iloc[::-1]
for idx, (_, row) in enumerate(top_3.iterrows(), 1):
    print(f"{idx}. {row['Feature']:15s} -> {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

# Create visualization
print("\n[*] Creating visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Color gradient from light to dark blue
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_importance_df)))

# Create horizontal bar chart
bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)

# Styling
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Importances - Random Forest (Flight Delay)', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save figure with lower DPI
output_file = 'feature_importance.png'
print(f"[*] Saving figure to '{output_file}'...")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"[OK] Figure saved: {output_file}")

# Display summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total Features: {len(feature_importance_df)}")
print(f"Most Important: {feature_importance_df.iloc[-1]['Feature']} ({feature_importance_df.iloc[-1]['Importance']:.4f})")
print(f"Least Important: {feature_importance_df.iloc[0]['Feature']} ({feature_importance_df.iloc[0]['Importance']:.4f})")
print(f"\nFull Feature Ranking:")
for idx, (_, row) in enumerate(feature_importance_df.iloc[::-1].iterrows(), 1):
    pct = int(row['Importance'] * 100)
    bar = '=' * (pct // 5) if pct > 0 else ''
    print(f"  {idx}. {row['Feature']:15s} {bar} {row['Importance']:.4f}")

print("\n" + "="*70)
print("[OK] ANALYSIS COMPLETE")
print("="*70)
