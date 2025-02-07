from sqlalchemy import create_engine, MetaData
from sqlalchemy.sql import text


def combine_tables():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        metadata = MetaData()
        metadata.reflect(bind=engine)

        query = """
        ALTER TABLE customers
        ADD COLUMN category_id BIGINT,
        ADD COLUMN category_code VARCHAR(255),
        ADD COLUMN brand VARCHAR(255);

        UPDATE customers c
        SET
            category_id = i.category_id,
            category_code = i.category_code,
            brand = i.brand
        FROM item i
        WHERE c.product_id = i.product_id;
        """

        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()

        print("Successfully combined 'customers' with 'items'.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    combine_tables()
