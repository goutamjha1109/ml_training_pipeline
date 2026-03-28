from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.client import MlflowClient
from dotenv import load_dotenv

from logger import logger
from data_loader import load_and_split_data, save_splits, build_preprocessor, save_pipeline
from utils import get_train_args, load_schema, load_params
from sklearn.pipeline import Pipeline


load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent



def build_pipeline(schema, model_params):
    logger.info("Building pipeline for the process ......")
    preprocessor = build_preprocessor(schema)  # just defines the steps, not fitted yet
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(**model_params))
    ])
    return pipeline

def get_or_create_experiment(name):
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)

    if exp:
        if exp.lifecycle_stage == "deleted":
            client.restore_experiment(exp.experiment_id)
        return exp.experiment_id
    else:
        return mlflow.create_experiment(name)


def main():
    args = get_train_args()
    schema = load_schema(args.schema)
    params = load_params(args.params)
    model_params = params["model"]
    paths = params["paths"]
    DATA_PATH       = Path(paths["raw_data"])
    PROCESSED_PATH  = Path(paths["processed_data"])
    ARTIFACTS_PATH  = Path(paths["artifacts"])
    # METRICS_PATH    = Path(paths["metrics"])
    # COMPARISON_PATH = Path(paths["comparison"])

    split = load_and_split_data(DATA_PATH, schema, params)
    pipeline = build_pipeline(schema, model_params)
    save_splits(split.X_train, split.X_test, split.y_train, split.y_test, PROCESSED_PATH)
    exp_id = get_or_create_experiment("churn-prediction")
    mlflow.set_experiment(experiment_id=exp_id)
    # mlflow.set_experiment("churn-prediction")
    print(mlflow.get_tracking_uri())

    # mlflow experiment tracking
    with mlflow.start_run():
        mlflow.log_params(model_params)
        logger.info("Starting training...")
        pipeline.fit(split.X_train, split.y_train)
        logger.info("Training completed.")
        y_pred = pipeline.predict(split.X_test)
        y_prob = pipeline.predict_proba(split.X_test)[:,1]
        accuracy = accuracy_score(split.y_test, y_pred)
        roc_auc = roc_auc_score(split.y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(pipeline, 
                                 name="churn_prediction",
                                 registered_model_name="churn-prediction-model")
        # Register to Model Registry
        # mlflow.register_model(
        #     model_uri=f"runs:/{mlflow.active_run().info.run_id}/churn_prediction",
        #     name="churn-prediction-model",
            
        # )
        logger.info(f"Logged — accuracy: {accuracy:.4f}, roc_auc: {roc_auc:.4f}")
    # 5. Save pipeline + target encoder
    save_pipeline(pipeline, split.le_target, ARTIFACTS_PATH)

if __name__ == "__main__":
    main()