from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import pandas as pd

def charts():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        SELECT 
            DATE(event_time) as event_date,
            COUNT(DISTINCT user_id) as num_customers,
            SUM(price) / 1e6 as total_sales_millions,
            SUM(price) / NULLIF(COUNT(DISTINCT user_id), 0) as avg_spend_per_customer
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY event_date
        ORDER BY event_date;
        """

        query_month = """
        SELECT 
            DATE_TRUNC('month', event_time) as event_month,
            SUM(price) / 1e6 as total_sales_millions
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY event_month
        ORDER BY event_month;
        """

        with engine.connect() as connection:
            res = connection.execute(text(query))
            month_res = connection.execute(text(query_month))
            data = res.fetchall()
            month_data = month_res.fetchall()

        df = pd.DataFrame(data, columns=['event_date', 'num_customers', 'total_sales_millions', 'avg_spend_per_customer'])
        month_df = pd.DataFrame(month_data, columns=['event_month', 'total_sales_millions'])

        # df['event_date'] = pd.to_datetime(df['event_date']).dt.strftime('%b')
        # month_df['event_month'] = pd.to_datetime(month_df['event_month']).dt.strftime('%b')

        plt.figure(figsize=(6, 12))

        plt.subplot(3, 1, 1)
        plt.plot(df['event_date'], df['num_customers'], color='blue', label="Number of Customers")
        plt.ylabel("Number of customers")
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.bar(month_df['event_month'], month_df['total_sales_millions'], color='orange', label="Total Sales (in Millions)", width = 28)
        plt.ylabel("Total sales in millions of ₳")
        plt.grid(axis='y')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(df['event_date'], df['avg_spend_per_customer'], color='green', label="Average Spend/Customer")
        plt.fill_between(df['event_date'], df['avg_spend_per_customer'], color='green', alpha=0.3)
        plt.ylabel("Average spend/customer in ₳")
        plt.grid()
        plt.legend()

        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    charts()
