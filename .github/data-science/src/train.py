# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument(
        '--max_depth', type=int, default=None,
        help='Max depth of each tree'
    )
    parser.add_argument(
        "--n_estimators", type=int, default=100,
        help="Number of trees in the forest"
    )
    parser.add_argument("--model_output", type=str, help="Path of output model")
 
    #Added to fix the error
    parser.add_argument('--criterion',
                        type=str,
                        default='squared_error',
                        help="Splitting metric for RandomForest (e.g. 'squared_error', 'absolute_error', etc.)")



    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.  
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  

    # Load data
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df  = pd.read_csv(select_first_file(args.test_data))

    y_train = train_df["price"].values
    X_train = train_df.drop("price", axis=1).values

    y_test  = test_df["price"].values
    X_test  = test_df.drop("price", axis=1).values

    # Initialize and train Random Forest regressor
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)

    # Evaluate with regression metrics
    mse = mean_squared_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²:  {r2:.4f}")

    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2",  r2)

    # Save the model
    mlflow.sklearn.save_model(rf_model, args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    os.makedirs("./outputs", exist_ok=True)

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

