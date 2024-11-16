import boto3
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def init_s3_client():
    """Initializes and returns an S3 client."""
    return boto3.client('s3')


def read_csv_from_s3(s3_client, bucket_name, file_key):
    """
    Reads a CSV file from an S3 bucket and returns it as a Pandas DataFrame.
    """
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(obj['Body'])


def split_data(series, train_ratio=0.7):
    """
    Splits the series into training and testing sets based on the train ratio.
    """
    train_size = int(len(series) * train_ratio)
    train_set = series[:train_size]
    test_set = series[train_size:]
    return train_set, test_set


def fit_arima_model(series, order=(1, 1, 1)):
    """
    Fits an ARIMA model to the given series.
    """
    model = ARIMA(series, order=order)
    return model.fit()


def evaluate_model(test_set, predicted_values):
    """
    Evaluates the model performance using MAE and RMSE.
    """
    mae = mean_absolute_error(test_set, predicted_values)
    rmse = np.sqrt(mean_squared_error(test_set, predicted_values))
    return mae, rmse


def main():
    # S3 details
    bucket_name = 'data-engineering-minhchef'
    file_key = 'rawData/time-series/DailyDelhiClimateTest_Processed.csv'

    # Initialize S3 client and load data
    s3_client = init_s3_client()
    df = read_csv_from_s3(s3_client, bucket_name, file_key)

    # Extract temperature series
    if 'meantemp' not in df.columns:
        print("Error: 'meantemp' column not found in the dataset.")
        return
    temperature_series = df['meantemp']

    # Split data into training and testing sets
    train_set, test_set = split_data(temperature_series, train_ratio=0.7)
    print(f"Training set size: {len(train_set)}, Testing set size: {len(test_set)}")

    # Fit ARIMA model on training data
    print("Fitting ARIMA model...")
    fitted_model = fit_arima_model(train_set, order=(1, 1, 1))
    print("ARIMA model fitted successfully.")

    # Forecast the testing set
    print("Generating forecast for the test set...")
    forecast_steps = len(test_set)
    forecast = fitted_model.forecast(steps=forecast_steps)
    print("Forecast complete.")

    # Evaluate the model
    mae, rmse = evaluate_model(test_set, forecast)
    print(f"Model Evaluation:\nMean Absolute Error (MAE): {mae}\nRoot Mean Square Error (RMSE): {rmse}")

    # Compare actual vs predicted values
    comparison = pd.DataFrame({
        "Actual": test_set.values,
        "Predicted": forecast
    })
    print("Comparison of Actual vs Predicted values:")
    print(comparison)


if __name__ == "__main__":
    main()
