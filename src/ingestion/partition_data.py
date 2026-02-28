import pandas as pd
from pathlib import Path

def partition_big_data():

    #define path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    INPUT_FILE = BASE_DIR / "data" / "raw" / "feature_usage_synthetic.csv"
    OUTPUT_DIR = BASE_DIR / "data" / "raw" / "daily_logs"

    # create output directory if not exists
    OUTPUT_DIR.mkdir(parents=True ,exist_ok=True)

    print(f"reading data for partitioning...")
    df = pd.read_csv(INPUT_FILE)


    #iterate through uniqe dates and save each as a separate file
    unique_dates = df['usage_date'].unique()


    print(f"partitioning data into {len(unique_dates)} daily files...")
    for date in unique_dates:
        daily_df = df[df["usage_date"] == date]
        file_name = f"logs_{date}.csv"
        daily_df.to_csv(OUTPUT_DIR / file_name , index=False)

    print(f"success! Daily logs created in {OUTPUT_DIR}")


if __name__ == "__main__":
    partition_big_data()