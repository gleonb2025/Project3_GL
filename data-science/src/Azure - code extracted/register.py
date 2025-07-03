
import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path

mlflow.start_run()  # Starting the MLflow experiment run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained model")
    args = parser.parse_args()

    # Load the trained Random Forest Regressor
    model = mlflow.sklearn.load_model(args.model)

    print("Registering the best trained model for used cars Random Forest Regressor")

    # Log & register under a new artifact path
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_cars_rf_regressor_model",
        artifact_path="random_forest_used_cars"
    )


    mlflow.end_run()

if __name__ == "__main__":
    main()
