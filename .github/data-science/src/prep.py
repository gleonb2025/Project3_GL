# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--used_cars_data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, help="Path to save train data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    args = parser.parse_args()

    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- WRITE YOUR CODE HERE -------

    # Step 1: Perform label encoding to convert categorical features into numerical values for model compatibility.  
    # Step 2: Split the dataset into training and testing sets using train_test_split with specified test size and random state.  
    # Step 3: Save the training and testing datasets as CSV files in separate directories for easier access and organization.  
    # Step 4: Log the number of rows in the training and testing datasets as metrics for tracking and evaluation.  

    print("Columns on disk:", df.columns.tolist())

    # Step 1: One-hot encode the 'Segment' categorical feature
    df = pd.get_dummies(
        df,
        columns=['Segment'],
        prefix='Segment',
        drop_first=True  # keeps just one dummy column (e.g. Segment_rural)
    )
    logging.info(f"Transformed data preview:\n{df.head()}")

    # Step 2: Split data
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_train_ratio,
        random_state=42
    )

    # Step 3: Save train/test to CSV
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_path = Path(args.train_data) / "data.csv"
    test_path  = Path(args.test_data)  / "data.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    logging.info(f"Saved train data to {train_path}")
    logging.info(f"Saved test data to  {test_path}")

    # Step 4: Log metrics to MLflow
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("test_rows",  len(test_df))


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.used_cars_data}",  # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
