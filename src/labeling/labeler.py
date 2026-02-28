import pandas as pd
from sqlalchemy import text
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.databases.db_loader import get_engine , load_to_postgres


def apply_behavioral_labeling(df):

    # sort for time-series consistency
    df = df.sort_values(['account_id', 'usage_date'])


    # significant drop compared to 30d baseline
    df['is_risk'] = (df['usage_drop_rate'] < 0.85).astype(int)


    # checks next 7 days
    df['max_7d'] = df.groupby('account_id')['usage_drop_rate'].transform(
        lambda x: x.shift(-7).rolling(window=7 , min_periods=1).max()
    )

    # checks next 14 days
    df['max_14d'] = df.groupby('account_id')['usage_drop_rate'].transform(
        lambda x: x.shift(-14).rolling(window=14 , min_periods=1).max()
    )

    # checks next 30 days
    df['max_30d'] = df.groupby('account_id')['usage_drop_rate'].transform(
        lambda x: x.shift(-30).rolling(window=30 , min_periods=1).max()
    )


    # final labeling logic
    df['target'] = (
        (df['is_risk'] == 1) & 
        (
        (df['max_7d'] < 0.85) |   
        (df['max_14d'] < 0.90) |  
        (df['max_30d'] < 0.95)    
        )
    ).astype(int)


    df = df.drop(columns=['is_risk', 'max_7d' , 'max_14d' ,'max_30d'])
    return df



def run_labeling_pipeline():
    engine = get_engine()

    try : 
        print(f"Reading from processed_features.")
        query = "SELECT * FROM processed_features"
        df = pd.read_sql(text(query), engine)
        
        print("Applying behavioral labeling logic")
        labeled_df = apply_behavioral_labeling(df)
        
        # Save to the final training table (replace mode for a fresh start)
        print(f"Saving labeled dataset ({len(labeled_df)} rows) to DB...")
        load_to_postgres(labeled_df, "labeled_features")
        
        risk_count = labeled_df['target'].sum()
        print(f"\nIdentified {risk_count:,} risk points.")
        print(f"Class Balance: {risk_count/len(labeled_df):.2%} of data is labeled as Churn Risk.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        engine.dispose()
        print("Database connection closed.")



if __name__ == "__main__":
    run_labeling_pipeline()