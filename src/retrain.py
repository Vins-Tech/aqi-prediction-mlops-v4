import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import os
import dagshub

# Load env
load_dotenv()

# DagsHub setup
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

# REMOVE local MLflow URI
# mlflow.set_tracking_uri(...) ❌

FEATURES = joblib.load("artifacts/selected_features.joblib")
TARGET = "aqipm25"
MODEL_NAME = "aqi_gbm_model"


def load_training_data():
    print("Loading training data...")
    df = pd.read_csv("data/latest.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Training data shape: {df.shape}")
    return df


def train(df):
    print("Training GradientBoostingRegressor...")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "random_state": 42
    }

    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return model, params, rmse, r2


def retrain():
    #  Only keep experiment (DagsHub handles URI)
    mlflow.set_experiment("aqi_prediction")

    df = load_training_data()

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        model, params, rmse, r2 = train(df)

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("training_rows", len(df))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"Model logged to MLflow (DagsHub).")
        print(f"Run ID: {run.info.run_id}")
        print(f"RMSE: {rmse:.4f} | R2: {r2:.4f}")

    return rmse, r2


if __name__ == "__main__":
    retrain()
