from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def customer_segmentation():
    try:
        engine = create_engine("postgresql://ndesprez:msp@localhost:5432/piscineds")

        query = """
        SELECT user_id, 
               COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count,
               MAX(event_time) AS last_purchase_date
        FROM customers
        GROUP BY user_id;
        """

        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

        df['days_since_last_purchase'] = (pd.Timestamp.now() - df['last_purchase_date']).dt.days

        X = df[['purchase_count', 'days_since_last_purchase']].values

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)

        cluster_centers = kmeans.cluster_centers_

        cluster_labels = {
            np.argmin(cluster_centers[:, 0]): "Inactive Customers",
            np.argmin(cluster_centers[:, 1]): "New Customers",
            np.argsort(cluster_centers[:, 0])[-2]: "Silver Customers",
            np.argmax(cluster_centers[:, 0]): "Gold Customers"
        }

        df['cluster_label'] = df['cluster'].map(cluster_labels)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        df['cluster_label'].value_counts().plot.pie(ax=ax1, autopct='%1.1f%%', startangle=90, 
                                                    colors=["gold", "silver", "red", "green"])
        ax1.set_title("Customer Segmentation Distribution")
        ax1.set_ylabel("")

        colors = {"Inactive Customers": "red", "New Customers": "green", 
                  "Silver Customers": "silver", "Gold Customers": "gold"}

        for cluster, label in cluster_labels.items():
            cluster_data = df[df['cluster'] == cluster]
            ax2.scatter(cluster_data['purchase_count'], cluster_data['days_since_last_purchase'], 
                        label=label, alpha=0.7, color=colors[label], edgecolors='black', s=50)

        ax2.set_xlabel("Number of Purchases (Actual)")
        ax2.set_ylabel("Days Since Last Purchase (Recency)")
        ax2.set_title("Customer Segmentation: Purchases vs. Recency")
        ax2.invert_yaxis()
        ax2.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    customer_segmentation()
