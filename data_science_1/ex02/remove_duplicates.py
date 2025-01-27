from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text


def remove_duplicates():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        metadata = MetaData()
        metadata.reflect(bind=engine)

        query_1 = """
        DELETE FROM customers_test
        WHERE ctid NOT IN (
            SELECT MIN(ctid)
            FROM customers_test
            GROUP BY user_id, event_time, event_type, product_id, price, user_session
        );
        """

        query_2 = """
        DELETE FROM customers_test c1
        USING customers_test c2
        WHERE c1.user_id = c2.user_id
        AND c1.event_time = c2.event_time + INTERVAL '1 second'
        AND c1.ctid < c2.ctid;
        """

        with engine.connect() as connection:
            connection.execute(text(query_1))
            connection.execute(text(query_2))
            connection.commit()

        print("Removed duplicate rows (including 1-second interval duplicates).")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    remove_duplicates()
