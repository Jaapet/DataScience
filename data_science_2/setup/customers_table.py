from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text
from colorama import Fore


def join_tables():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        metadata = MetaData()
        metadata.reflect(bind=engine)

        tables = [table for table in metadata.tables.keys() if table.startswith("data_202")]

        join_query = None
        if tables:
            join_query = f"SELECT * FROM {tables[0]}"
        for table in tables[1:]:
            join_query += f" UNION ALL SELECT * FROM {table}"
        query = f"CREATE TABLE IF NOT EXISTS customers AS ({join_query});"

        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        print(Fore.GREEN + "Successfully created 'customers' table")

    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}")


if __name__ == "__main__":
    join_tables()
