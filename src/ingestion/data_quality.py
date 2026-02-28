import pandas as pd
import os
from prefect import task

RAW_DIR = "data/raw/daily_logs/"
PROCESSED_DIR = "data/processed/ingested_logs/"
QUARANTINE_DIR = "data/quarantine/"

@task(name="Validate Data Quality")
def validate_data_quality(df,date_str):

    print(f"Starting validation for batch: {date_str}")
    file_name = f"logs_{date_str}.csv"

    # Define expected schema
    expected_columns = {
        'account_id': 'object',
        'usage_date': 'object',
        'feature_name': 'object',
        'usage_count': 'int64',
        'usage_duration_secs': 'int64',
        'error_count': 'int64',
    }


    # Schema Validation
    missing_cols = set(expected_columns.keys() - set(df.columns))
    if missing_cols: 
        print(f"Missing columns: {missing_cols}")

    # Missing value check
    critical_fields = ['account_id' , 'usage_date' , 'feature_name']
    null_mask = df[critical_fields].isnull().any(axis=1)
    df_faulty_nulls = df[null_mask]

    # Timestamp anomalies
    df['usage_date_dt'] = pd.to_datetime(df['usage_date'])
    wrong_date_mask = df['usage_date_dt'].dt.strftime('%Y-%m-%d') != date_str
    df_faulty_dates = df[wrong_date_mask]

    # Logical Checks
    negative_mask = (
        (df['usage_count'] < 0) | 
        (df['usage_duration_secs'] < 0) | 
        (df['error_count'] < 0)
    )
    df_faulty_logic = df[negative_mask]


    # Feature name check
    mask_invalid_feature = df['feature_name'].isnull() | ~df['feature_name'].apply(lambda x: isinstance(x,str))
    df_faulty_features = df[mask_invalid_feature]

    
    # Combine all faulty records
    faulty_df = pd.concat([
        df_faulty_nulls, 
        df_faulty_dates, 
        df_faulty_logic, 
        df_faulty_features
    ]).drop_duplicates()

    clean_df = df.drop(faulty_df.index)

    for frame in [clean_df, faulty_df]:
        if 'usage_date_dt' in frame.columns:
            frame.drop(columns=['usage_date_dt'], inplace=True)

    # Logging faulty records separately
    if not faulty_df.empty:
        if not os.path.exists(QUARANTINE_DIR):
            os.makedirs(QUARANTINE_DIR)
        q_path = os.path.join(QUARANTINE_DIR, f"faulty_{date_str}.csv")
        faulty_df.to_csv(q_path, index=False)
        print(f" {len(faulty_df)} faulty records moved to quarantine: {q_path}")

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    processed_path = os.path.join(PROCESSED_DIR, file_name)
    clean_df.to_csv(processed_path, index=False)
    print(f"Clean records saved to: {processed_path}")

    raw_path = os.path.join(RAW_DIR, file_name)
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"Original raw file removed: {raw_path}")

    print(f"Validation finished. Clean records : {len(clean_df)}")

    return clean_df