import sys
import os
import pandas as pd
from prefect import task, flow
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ingestion.data_quality import validate_data_quality
from src.databases.db_loader import load_to_postgres
from src.features.feature_pipeline import extract_and_engineering_features , save_features


@task(retries=3, retry_delay_seconds=10)
def get_daily_file_path(directory, date_str):
    """
    Finds the file with 'logs_' prefix.
    """
    file_name = f"logs_{date_str}.csv"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        return file_path
    else:
        print(f"Missing data for date: {date_str}")
        raise FileNotFoundError(f"File not found: {file_path}")


@task
def ingest_daily_data(file_path):
    """
    Reads the CSV file into a DataFrame.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} events from {file_path}")
    return df


@flow(name="Daily SaaS Event Ingestion")
def run_ingestion_pipeline(target_date: str):
    
    print(f"Starting pipeline for {target_date}")

    raw_dir = "data/raw/daily_logs/"

    # get the file path
    file_path = get_daily_file_path(raw_dir, target_date)

    # read data
    raw_df = ingest_daily_data(file_path)

    # data quality check
    clean_df = validate_data_quality(raw_df, target_date)

    # Load to DB
    load_to_postgres(clean_df, "feature_usage")
    



if __name__ == "__main__":
    
    start_date = "2023-01-01"
    end_date = "2024-12-30"

    date_list = pd.date_range(start = start_date , end=end_date).strftime('%Y-%m-%d')

    print(f"starting ingestion for {len(date_list)} days...")

    print(f"--- Phase 1: Bulk Ingestion ({len(date_list)} days) ---")

    for date in date_list:
        try:
            run_ingestion_pipeline(date)
        except Exception:
            continue 

    print("\n--- Phase 2: Single-Pass Feature Engineering ---")
    
    featured_df = extract_and_engineering_features()
    save_features(featured_df)

    print("İngestion completed.")    