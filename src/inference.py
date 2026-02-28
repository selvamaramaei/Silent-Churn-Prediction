import pandas as pd
import xgboost as xgb
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.databases.db_loader import get_engine

def run_inference():
    engine = get_engine()
    
    # 1. Load the model
    model_path = os.path.join("models", "silent_churn_v1.json")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"[INFO] Model loaded successfully: {model_path}")

    # 2. Fetch the most recent processed features
    query = "SELECT * FROM processed_features ORDER BY usage_date DESC LIMIT 1000"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("[WARNING] No data found in processed_features table.")
        return

    # 3. Select model features
    features = ['daily_usage', 'usage_7d_avg', 'usage_30d_avg', 'usage_drop_rate']
    X = df[features]
    
    print(f"Calculating risk scores for {len(df)} users...")
    
    # Probability of being churn (class 1)
    probs = model.predict_proba(X)[:, 1]
    
    # 'Safe' threshold determined during analysis
    threshold = 0.85
    df['risk_score'] = probs
    df['is_risk'] = (probs >= threshold).astype(int)

    # Filter high-risk users
    at_risk_users = df[df['is_risk'] == 1].sort_values(by='risk_score', ascending=False)
    
    print("\n" + "="*50)
    print(f"SILENT CHURN RISK REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*50)
    
    if at_risk_users.empty:
        print("Everything looks stable! No immediate risks detected.")
    else:
        print(f"Detected {len(at_risk_users)} users at high risk (Threshold: {threshold}):")
        print(
            at_risk_users[['account_id', 'usage_date', 'risk_score']]
            .head(10)
            .to_string(index=False)
        )
        
    print("="*50 + "\n")


if __name__ == "__main__":
    run_inference()