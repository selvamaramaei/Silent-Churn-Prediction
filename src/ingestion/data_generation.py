import pandas as pd
import numpy as np
from datetime import timedelta,datetime
from pathlib import Path

# Configuration for reproducibility
np.random.seed(42)


def generate_synthetic_usage(accounts_df, feature_usage_df ,start_date_str="2023-01-01" , days=730):

    """
    generates stockastic time-series usage data for given accounts.
    """

    start_date = datetime.strptime(start_date_str,"%Y-%m-%d")
    all_events = []

    # feature list 
    features = feature_usage_df['feature_name'].unique().tolist()

    for _ , row in accounts_df.iterrows():
        acc_id = row['account_id']
        is_churner = row['churn_flag'] # we use existing flag as a base guide

        critical_feature = np.random.choice(features)

        # 1. individual personality assignment
        # base daily activity : mean events per dat
        base_lambda = np.random.uniform(2,8)

        # 2. Assign random decay parameters per churners
        decay_rate = 0 
        decay_start_day = 999
        
        if is_churner:
            # randomize when they start to quit (between day 150 and 300)
            decay_start_day = np.random.randint(400,650)
            # randomize how fast they quit 
            decay_rate = np.random.normal(0.015 , 0.005)

        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)

            # 3. calculate dynamic lambda (activity rate)
            current_lambda = apply_advanced_stochastic_logic(
                is_churner, day_offset, base_lambda, decay_start_day
            )

            # 4. generate daily events with poisson noise
            daily_count = np.random.poisson(current_lambda)    

            for _ in range(daily_count):

                feature = np.random.choice(features)
                # sudden drop simulation for specific feature
                if is_churner and day_offset > (decay_start_day + np.random.randint(10, 30)):
                    if feature == critical_feature:
                        if feature == critical_feature and np.random.rand() > 0.2:
                            continue


                all_events.append({
                    'account_id' : acc_id,
                    'usage_date': current_date,
                    'feature_name': feature,
                    'usage_count': 1,
                    'usage_duration_secs': np.random.randint(30, 600),
                    'error_count': np.random.choice([0, 1, 2], p=[0.9, 0.08, 0.02])
                })    
   
    return pd.DataFrame(all_events)


# Logic for advanced randomness
def apply_advanced_stochastic_logic(is_churner, day_offset, base_lambda, decay_start_day):
    """
    Prevents model memorization by adding false alarms and complex decay patterns.
    """
    # 1. False Alarm for Active Users (The "Vacation" Effect)
    # A temporary dip for non-churners to prevent the model from over-relying on simple activity drops
    if not is_churner and 100 < day_offset < 114: # A 2-week dip
        return base_lambda * 0.2 

    # 2. Complex Decay for Churners
    if is_churner and day_offset > decay_start_day:
        # Instead of a constant rate, we use a rate that varies slightly per day
        # Using a Gamma distribution adds natural 'hesitation' in the churn process
        stochastic_decay = np.random.gamma(shape=2.0, scale=0.01)
        t = day_offset - decay_start_day
        return base_lambda * np.exp(-stochastic_decay * t)
    
    return base_lambda


if __name__ == "__main__":
    # Define paths relative to this script
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"

    # 1. Load the Master Data (Accounts)
    print("Loading account data...")
    try:
        df_accounts = pd.read_csv(DATA_RAW_DIR / "accounts.csv")
        df_feature_usage = pd.read_csv(DATA_RAW_DIR / "feature_usage.csv")
    except FileNotFoundError:
        print(f"Error: files not found in {DATA_RAW_DIR}")
        exit()

    # 2. Generate Synthetic Usage
    print("Generating synthetic usage logs (this may take a moment)...")
    synthetic_usage = generate_synthetic_usage(df_accounts,df_feature_usage)

    # 3. Save to data/raw/
    output_path = DATA_RAW_DIR / "feature_usage_synthetic.csv"
    synthetic_usage.to_csv(output_path, index=False)

    print(f"Success! Generated {len(synthetic_usage)} events.")
    print(f"File saved to: {output_path}")
    print("\n--- Preview ---")
    print(synthetic_usage.head())