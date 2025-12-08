# pipelines/train_pipeline.py (Prefect exemple)
from prefect import flow, task
from src.data_preprocessing import preprocess_data
from src.train_model import train_model

@task
def preprocess_task():
    return preprocess_data("data/raw/secom.csv")

@task
def train_task(df):
    train_model(df)

@flow
def mlops_flow():
    df = preprocess_task()
    train_task(df)

if __name__ == "__main__":
    mlops_flow()
