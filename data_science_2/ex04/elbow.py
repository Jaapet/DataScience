from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def elbow():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        SELECT user_id, COUNT(*) AS purchases
        FROM customers
        WHERE event_type = 'purchase'
        GROUP BY user_id
        HAVING COUNT(*) < 30
        ORDER BY purchases DESC;
        """

        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        X = df[['purchases']].values

        inertia = []
        K_range = range(1, 11)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(K_range, inertia, marker='o', linestyle='--')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia (WCSS)")
        plt.title("Elbow Method for Optimal K")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    elbow()