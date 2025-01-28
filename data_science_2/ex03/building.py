from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import numpy as np

def orders():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query_order = """
        SELECT COUNT(*)
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id
        """

        query_spend = """
        SELECT SUM(price)
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id;
        """

        with engine.connect() as connection:
            orders_result = connection.execute(text(query_order))
            orders_data = [row[0] for row in orders_result.fetchall()]
            spending_result = connection.execute(text(query_spend))
            spending_data = [row[0] for row in spending_result.fetchall()]

        orders_bins = np.linspace(0, 40, 6)
        spending_bins = [-25, 25, 75, 125, 175, 225]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.hist(orders_data, bins=orders_bins, color='skyblue', edgecolor='black')
        plt.xlim(0, 40)
        plt.xticks(np.linspace(0, 40, 5))
        plt.xlabel("Commands")
        plt.ylabel("Customers")
        plt.grid(axis='y')

        plt.subplot(2, 1, 2)
        plt.hist(spending_data, bins=spending_bins, color='orange', edgecolor='black', align='mid')
        plt.xlim(-25, 225)
        plt.xticks(np.linspace(0, 200, 5))
        plt.xlabel("Total Spent")
        plt.ylabel("Customers")
        plt.grid(axis='y')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    orders()
