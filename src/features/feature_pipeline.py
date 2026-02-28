import pandas as pd
from sqlalchemy import text
from src.databases.db_loader import get_engine , load_to_postgres
from prefect import task


@task(name='Extract and engineering features', retries=2)
def extract_and_engineering_features():
    
    engine= get_engine()

    query = """
    SELECT account_id, usage_date , SUM(usage_count) as daily_usage
    FROM feature_usage
    GROUP BY  account_id , usage_date ORDER BY usage_date
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query) , conn)


    # rolling windows for trend analysis
    df['usage_7d_avg'] = df.groupby('account_id')['daily_usage'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df['usage_30d_avg'] = df.groupby('account_id')['daily_usage'].transform(
        lambda x: x.rolling(30 , min_periods=30).mean()
    )

    # ratio of current week to month
    df['usage_drop_rate'] = df['usage_7d_avg'] / (df['usage_30d_avg'] + 1e-9)

    return df



@task(name='Save features to DB')
def save_features(df):

    load_to_postgres(df , "processed_features")



  


