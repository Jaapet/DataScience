from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import pandas as pd

def prices():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        SELECT price
        FROM customers
        WHERE event_type = 'purchase';
        """

        query_basket = """
        SELECT user_id, SUM(price) as basket_price
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id;
        """

        with engine.connect() as connection:
            res = connection.execute(text(query))
            prices = [row[0] for row in res.fetchall()]
            basket_res = connection.execute(text(query_basket))
            basket_prices = [row[1] for row in basket_res.fetchall()]

        prices_df = pd.Series(prices)
        basket_prices_df = pd.Series(basket_prices)

        count_price = prices_df.count()
        mean_price = prices_df.mean()
        median_price = prices_df.median()
        std_price = prices_df.std()
        min_price = prices_df.min()
        max_price = prices_df.max()
        first_quartile = prices_df.quantile(0.25)
        second_quartile = median_price
        third_quartile = prices_df.quantile(0.75)

        print(f"Count of Price:              {count_price:.6f}")
        print(f"Mean Price:                  {mean_price:.6f}")
        print(f"Median Price:                {median_price:.6f}")
        print(f"Standard Deviation of Price: {std_price:.6f}")
        print(f"Min Price:                   {min_price:.6f}")
        print(f"Max Price:                   {max_price:.6f}")
        print(f"First Quartile (Q1):         {first_quartile:.6f}")
        print(f"Second Quartile (Q2):        {second_quartile:.6f}")
        print(f"Third Quartile (Q3):         {third_quartile:.6f}")

        plt.figure(figsize=(6, 12))

        plt.subplot(3, 1, 1)
        plt.boxplot(prices_df, vert=False, patch_artist=True)
        plt.title("Prices")
        plt.xlabel("Price")

        plt.subplot(3, 1, 2)
        plt.boxplot(prices_df, vert=False, patch_artist=True)
        plt.title(f"Prices")
        plt.xlabel("Price")
        plt.xlim(-1, 13)

        plt.subplot(3, 1, 3)
        plt.boxplot(basket_prices_df, vert=False, patch_artist=True)
        plt.title("Average Basket Price")
        plt.xlabel("Basket Price")
        plt.xlim(0, 150)

        plt.tight_layout(pad=5.0)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    prices()