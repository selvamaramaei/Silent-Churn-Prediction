import pandas as pd
import joblib
import os
import sys
from datetime import datetime

# Path setup for db_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.databases.db_loader import get_engine

def run_rf_inference():
    engine = get_engine()
    
    # Load RF Model
    model_path = os.path.join("models", "rf_silent_churn_v1.joblib")
    if not os.path.exists(model_path):
        print(f"[ERROR] RF Model not found at: {model_path}")
        return
        
    rf_model = joblib.load(model_path)
    print(f"[INFO] RF Model loaded successfully: {model_path}")

    # Fetch recent data
    query = "SELECT * FROM processed_features ORDER BY usage_date DESC LIMIT 1000"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("[WARNING] No data found in processed_features.")
        return

    # Select features
    features = ['daily_usage', 'usage_7d_avg', 'usage_30d_avg', 'usage_drop_rate']
    X = df[features]
    
    # Prediction logic
    probs = rf_model.predict_proba(X)[:, 1]
    threshold = 0.50 # Default threshold for RF evaluation
    df['risk_score'] = probs
    df['is_risk'] = (probs >= threshold).astype(int)

    # Filter high-risk users
    at_risk = df[df['is_risk'] == 1].sort_values(by='risk_score', ascending=False)
    
    print("\n" + "="*50)
    print(f" RANDOM FOREST RISK REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*50)
    
    if at_risk.empty:
        print("Everything looks stable! No immediate risks detected.")
    else:
        print(f"Detected {len(at_risk)} users at high risk (Threshold: {threshold}):")
        print(at_risk[['account_id', 'usage_date', 'risk_score']].head(10).to_string(index=False))
        
    print("="*50 + "\n")

if __name__ == "__main__":
    run_rf_inference()