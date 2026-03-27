import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os
import dagshub

load_dotenv()

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

if DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME and DAGSHUB_USER_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_USER_TOKEN

    dagshub.init(
        repo_owner=DAGSHUB_REPO_OWNER,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )

MODEL_NAME = "aqi_gbm_model"


def get_production_rmse(client):
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            print("No Production model found.")
            return None, None

        prod_version = versions[0]
        run_id = prod_version.run_id
        run = client.get_run(run_id)
        rmse = run.data.metrics["rmse"]
        print(f"Current Production model: version {prod_version.version}, RMSE: {rmse:.4f}")
        return rmse, prod_version.version

    except Exception as e:
        print(f"Error fetching Production model: {e}")
        return None, None


def get_latest_staging_rmse(client):
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
        if not versions:
            print("No new model found in registry.")
            return None, None

        latest_version = max(versions, key=lambda v: int(v.version))
        run_id = latest_version.run_id
        run = client.get_run(run_id)
        rmse = run.data.metrics["rmse"]
        print(f"New model: version {latest_version.version}, RMSE: {rmse:.4f}")
        return rmse, latest_version.version

    except Exception as e:
        print(f"Error fetching new model: {e}")
        return None, None


def evaluate():
    client = MlflowClient()

    print("Comparing models...")
    print("---")

    prod_rmse, prod_version = get_production_rmse(client)
    new_rmse, new_version = get_latest_staging_rmse(client)

    if new_rmse is None:
        print("No new model to evaluate.")
        return

    if prod_rmse is None:
        print(f"No Production model exists. Promoting version {new_version} to Production.")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production"
        )
        print(f"Version {new_version} is now Production.")
        return

    print("---")
    print(f"Production RMSE: {prod_rmse:.4f}")
    print(f"New model RMSE:  {new_rmse:.4f}")
    print("---")

    if new_rmse < prod_rmse:
        print(f"New model is BETTER. Promoting version {new_version} to Production.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prod_version,
            stage="Archived"
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production"
        )
        print(f"Version {new_version} is now Production.")
        print(f"Version {prod_version} has been Archived.")

    else:
        print(f"New model is NOT better. Keeping version {prod_version} as Production.")

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Archived"
        )
        print(f"Version {new_version} has been Archived.")


if __name__ == "__main__":
    evaluate()