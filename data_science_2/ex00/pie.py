from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

def pie_chart():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        SELECT event_type, COUNT(*) as count
        FROM customers
        GROUP BY event_type;
        """

        with engine.connect() as connection:
            result = connection.execute(text(query))
            data = result.fetchall()

        event_types = [row[0] for row in data]
        counts = [row[1] for row in data]

        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=event_types, autopct='%1.1f%%', startangle=90)
        plt.title("Event Types Distribution")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    pie_chart()
