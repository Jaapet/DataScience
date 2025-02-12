import pandas as pd


def correlation_with_target(path: str):
    """
    Calculates the correlation between the target column ('knight') and all numerical features in the dataset.
    The 'knight' column is mapped to numerical values ('Sith' = 0, 'Jedi' = 1) to enable correlation analysis. 
    The Pearson correlation coefficient is used to measure the linear relationship between 'knight' and each feature.
    The result shows the strength and direction of the relationship,
    with values close to 1 or -1 indicating strong correlations, and values near 0 indicating weak or no correlation.
    """

    try:
        df = pd.read_csv(path)

        df['knight'] = df['knight'].map({'Sith': 0, 'Jedi': 1})

        corr = df.corr()

        corr = corr["knight"].drop("knight").abs()

        corr = corr.sort_values(ascending=False)

        print("Strongest correlations with target column '{}':".format("knight"))
        print(corr)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    correlation_with_target("../Train_knight.csv")
