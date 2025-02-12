import pandas as pd
import matplotlib.pyplot as plt


def histogram_test(path:str):
    try:
        df = pd.read_csv(path)

        numerical_columns = df.select_dtypes(include=["number"]).columns
        num_cols = len(numerical_columns)

        fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(15, 5 * ((num_cols // 3) + 1)))

        axes = axes.flatten() if num_cols > 1 else [axes]

        for i, column in enumerate(numerical_columns):
            axes[i].hist(df[column].dropna(), bins=30, color="green", alpha=0.4)
            axes[i].set_title(f"{column}", fontsize=10)

        plt.tight_layout(pad=0.0, h_pad=20.0, w_pad=0.0)
        plt.subplots_adjust(top=0.97, right=0.99, bottom=0.03)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


def histogram_train(path: str):
    try:
        df = pd.read_csv(path)

        numerical_columns = df.select_dtypes(include=["number"]).columns
        num_cols = len(numerical_columns)

        fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(15, 5 * ((num_cols // 3) + 1)))
        axes = axes.flatten() if num_cols > 1 else [axes]

        df_jedi = df[df["knight"] == "Jedi"]
        df_sith = df[df["knight"] == "Sith"]

        for i, column in enumerate(numerical_columns):
            axes[i].hist(df_jedi[column].dropna(), bins=30, color="blue", alpha=0.5, label="Jedi")
            axes[i].hist(df_sith[column].dropna(), bins=30, color="red", alpha=0.5, label="Sith")
            axes[i].set_title(f"{column}", fontsize=10)
            axes[i].legend()

        plt.tight_layout(pad=0.0, h_pad=20.0, w_pad=0.0)
        plt.subplots_adjust(top=0.97, right=0.99, bottom=0.03)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    histogram_test("../Test_knight.csv")
    histogram_train("../Train_knight.csv")
