import boto3
import pandas as pd
import io


# Constants
BUCKET_NAME = 'data-engineering-minhchef'
INPUT_FILE_KEY = 'rawData/time-series/DailyDelhiClimateTest.csv'
OUTPUT_FILE_KEY = 'rawData/time-series/DailyDelhiClimateTest_Processed.csv'


def init_s3_client():
    """
    Initialize the S3 client.
    Returns:
        boto3.client: S3 client instance.
    """
    return boto3.client('s3')


def read_csv_from_s3(s3_client, bucket_name, file_key):
    """
    Reads a CSV file from an S3 bucket and returns it as a Pandas DataFrame.
    Args:
        s3_client (boto3.client): The initialized S3 client.
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key (path) to the file in the S3 bucket.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return pd.read_csv(obj['Body'])
    except Exception as e:
        print(f"Error reading {file_key} from S3: {e}")
        raise


def save_df_to_s3(df, s3_client, bucket_name, file_key):
    """
    Saves a Pandas DataFrame as a CSV file to an S3 bucket.
    Args:
        df (pd.DataFrame): The DataFrame to save.
        s3_client (boto3.client): The initialized S3 client.
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key (path) to save the file in the S3 bucket.
    """
    try:
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)  # Ensure no index is added
        csv_buffer.seek(0)
        s3_client.upload_fileobj(csv_buffer, bucket_name, file_key)
        print(f"File successfully saved to S3 at {file_key}")
    except Exception as e:
        print(f"Error saving DataFrame to S3: {e}")
        raise


def main():
    """
    Main function to process the CSV file from S3 and save the processed version back to S3.
    """
    s3_client = init_s3_client()

    # Read the input CSV file
    df = read_csv_from_s3(s3_client, BUCKET_NAME, INPUT_FILE_KEY)
    print(f"Successfully read file from S3: {INPUT_FILE_KEY}")

    # Perform any processing on the DataFrame here (if needed)
    # Example: Process data (no processing applied in this example)
    # df = process_data(df)

    # Save the processed DataFrame back to S3
    save_df_to_s3(df, s3_client, BUCKET_NAME, OUTPUT_FILE_KEY)


if __name__ == "__main__":
    main()
