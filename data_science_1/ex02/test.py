from sqlalchemy import create_engine, text


def create_customers_test():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        CREATE TABLE IF NOT EXISTS customers_test AS 
        SELECT * FROM customers
        LIMIT 10000;
        """

        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        print("Successfully created 'customers_test' table with the first 10000 rows.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    create_customers_test()
