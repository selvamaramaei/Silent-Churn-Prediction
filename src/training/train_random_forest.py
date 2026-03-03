import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix
import os
import sys
import joblib


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.databases.db_loader import get_engine


def train_rf_model():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM labeled_features" , engine)


    features = ['daily_usage','usage_7d_avg' , 'usage_30d_avg','usage_drop_rate']
    X = df[features]
    y = df['target']

    X_train , X_test , y_train , y_test = train_test_split(
        X,y, test_size=0.2 , stratify=y
    )


    # Random Foreset model 

    print("Training random forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train,y_train)

    # evaluation
    y_pred = rf_model.predict(X_test)
    print(f"---Random forest performans report---")
    print(classification_report(y_test,y_pred))

    # save the model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(rf_model, os.path.join(model_dir, "rf_silent_churn_v1.joblib"))
    print(f"Model saved to {model_dir}/rf_silent_churn_v1.joblib")

if __name__ == "__main__":
    train_rf_model()