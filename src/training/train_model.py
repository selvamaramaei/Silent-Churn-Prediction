import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.databases.db_loader import get_engine
import joblib 



def train_model():
    engine = get_engine()

    # Fetch data
    print("Loading labeled data from DB")
    df = pd.read_sql("SELECT * FROM labeled_features" , engine)

    # set feature and target
    features = ['daily_usage' , 'usage_7d_avg' , 'usage_30d_avg' , 'usage_drop_rate']
    X = df[features]
    y = df['target']

    # Train-Test split
    X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # XGBoost Model
    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    print(f"Training XGBoost model with scale_pos_weight : {scale_weight: .2f}")

    model= xgb.XGBClassifier(
        n_estimators=100,
        max_depth = 6,
        learning_rate = 0.1,
        scale_pos_weight = scale_weight,
        use_label_encoder = False,
        eval_metric = 'logloss'
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save_model("models/silent_churn_v1.json")
    print("Model saved to models/silent_churn_v1.json")

if __name__ == "__main__":
    train_model()
    
