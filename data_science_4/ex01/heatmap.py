import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def plot_correlation_heatmap(path):
    """
    Reads a CSV file, computes correlations, and plots a heatmap.

    The Pearson correlation coefficient is used to measure the linear relationship between 'knight' and each feature.
    The result shows the strength and direction of the relationship,
    with values close to 1 or -1 indicating strong correlations, and values near 0 indicating weak or no correlation.
    """
    try:
        df = pd.read_csv(path)

        last_column = df.columns[-1]

        # Encode the categorical target column to
        # numeric values to enable correlation analysis.
        encoder = LabelEncoder()
        df[last_column] = encoder.fit_transform(df[last_column])

        correlation_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap="coolwarm")
        plt.title("Feature Correlation")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    plot_correlation_heatmap("../Train_knight.csv")
