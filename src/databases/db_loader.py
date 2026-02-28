import os
import pandas as pd 
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prefect import task


#load info from .env
load_dotenv()

# Database connetion info
DB_USER = os.getenv("DB_USER" , "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD" , "1234")
DB_HOST = os.getenv("DB_HOST" , "localhost")
DB_PORT = os.getenv("DB_PORT" , "5432")
DB_NAME = "Silent_churn_db"


def get_engine():
    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


@task(name="Load data to postgres")
def load_to_postgres(df: pd.DataFrame , tabel_name: str):
    # Writes dataframe to postgresql database 

    if df.empty:
        print("Warning : DataFrame is empty.")
        return 
    
    engine = get_engine()
    try : 
    
        print(f"Uploading {len(df)} rows to tabel : {tabel_name}")
        df.to_sql(tabel_name , engine , if_exists="append" , index=False)
        print("Data loaded to DB successfully")

    except Exception as e:
        print(f"Error loading to DB : {e}")
        raise e 
    
    finally:

        #This releases all socket resources
        engine.dispose()
        print("Engine connection pool disposed.")
    
