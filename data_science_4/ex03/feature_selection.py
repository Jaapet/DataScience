import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


def vif(df):
    """Compute VIF for each feature in the dataset.
    Variance Inflation Factor (VIF) measures how much
    a feature is correlated with other features in the dataset.
    It helps detect multicollinearity, which occurs when
    predictor variables are highly correlated,
    leading to unstable models.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif_data["Tolerance"] = 1 / np.array(vif_data["VIF"])
    return vif_data


def rm_high_vif(df, threshold=5):
    """Remove features iteratively until all VIF values are below the threshold.
    The threshold is the VIF value beyond which we decide
    a feature has too much multicollinearity and should be removed.
    """
    while True:
        vif_data = vif(df)
        max_vif = vif_data["VIF"].max()

        if max_vif < threshold:
            break

        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        df = df.drop(columns=[feature_to_remove])

    return vif_data


if __name__ == "__main__":
    try:

        df = pd.read_csv("../Train_knight.csv")
        #  Selects all columns except the last one (class)
        df = df.iloc[:, :-1]

        scaler = StandardScaler()
        standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        final_vif = rm_high_vif(standardized)

        print(final_vif.to_string(index=False, float_format="%.6f"))

    except Exception as e:
        print(f"Error: {e}")
