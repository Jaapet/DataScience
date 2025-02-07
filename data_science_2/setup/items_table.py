import pandas as pd
import os
from sqlalchemy import create_engine, MetaData, types
from colorama import Fore


def check_table(engine, table: str) -> bool:
    """
    Checks if a table exists in the database.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return table in metadata.tables


def create_table(path: str) -> None:
    """
    Loads data from a CSV file into a PostgreSQL table if it doesn't already exist.
    """
    try:
        # Create SQLAlchemy engine to connect to PostgreSQL database
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")
        # List all files in the directory
        for file in os.listdir(path):
            # Check if the file is a CSV file
            if file.endswith(".csv"):
                # Extract the name without the .csv extension
                table = os.path.splitext(file)[0]
                # Full path to the CSV file
                file_path = os.path.join(path, file)

            # Check if the table already exists in the database
            if not check_table(engine, table):
                print(Fore.YELLOW + f"Creating table {table}")
                # Read CSV file into pandas DataFrame
                data = pd.read_csv(file_path)
                # Define the column data types for the table
                columns = {
                    "product_id": types.Integer(),
                    "category_id": types.BigInteger(),
                    "category_code": types.String(length=255),
                    "brand": types.String(length=255)
                }
                # Write the DataFrame to the PostgreSQL table
                data.to_sql(table, engine, index=False, dtype=columns, if_exists='fail')
                print(Fore.GREEN + f"Table {table} created.")
            else:
                print(Fore.BLUE + f"Table {table} already exists.")

    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}")


if __name__ == "__main__":
    create_table("/goinfre/ndesprez/subject/item")
