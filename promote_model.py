from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "SECOM_Classifier"
latest_version = client.get_latest_versions(model_name)[-1].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print("✅ Model promoted to Production")
