import pandas as pd
from sklearn.model_selection import train_test_split

def split(path: str, ratio=0.8):
    """
    The 80-20 split is a common practice because:

    Enough data for training:   The model learns well with a larger dataset.
    Reliable validation:        The validation set is large enough to detect overfitting.
    Standard industry practice: Many ML tasks use this ratio.
    """
    try:
        df = pd.read_csv(path)

        train_df, val_df = train_test_split(df, test_size=(1 - ratio))

        train_df.to_csv("../Training_knight.csv", index=False)
        val_df.to_csv("../Validation_knight.csv", index=False)

        print(f"Training set: {len(train_df)} samples ({round(ratio * 100, 2)}%)")
        print(f"Validation set: {len(val_df)} samples ({round((1 - ratio) * 100, 2)}%)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    split("../Train_knight.csv")
