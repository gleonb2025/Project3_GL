
import mlflow
import argparse
import os
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.start_run()

os.makedirs("./outputs", exist_ok=True)

def select_first_file(path):
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
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

    mlflow.end_run()

if __name__ == "__main__":
    main()
