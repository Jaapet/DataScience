import pandas as pd
from sqlalchemy import create_engine, MetaData, types
from sqlalchemy.exc import SQLAlchemyError


def check_table(engine, table: str) -> bool:
    """
    Checks if a table exists in the database.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return table in metadata.tables


def create_table(path: str, table: str) -> None:
    """
    Loads data from a CSV file into a PostgreSQL table if it doesn't already exist.
    """
    try:
        # Create SQLAlchemy engine to connect to PostgreSQL database
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")
        # Check if the table already exists in the database
        if not check_table(engine, table):
            print(f"Creating table {table}")
            # Read CSV file into pandas DataFrame
            data = pd.read_csv(path)
            # Define the column data types for the table
            columns = {
                "event_time": types.DateTime(),
                "event_type": types.String(length=255),
                "product_id": types.Integer(),
                "price": types.Float(),
                "user_id": types.BigInteger(),
                "user_session": types.UUID(as_uuid=True)
            }
            # Write the DataFrame to the PostgreSQL table
            data.to_sql(table, engine, index=False, dtype=columns, if_exists='fail')
            print(f"Table {table} created.")

        else:
            print(f"Table {table} already exists.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_table("/goinfre/ndesprez/subject/customer/data_2023_jan.csv", "data_2023_jan")
