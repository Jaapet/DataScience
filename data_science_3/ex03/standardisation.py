import pandas as pd
import matplotlib.pyplot as plt

def standardize(path_1: str, path_2: str):
    try:
        df1 = pd.read_csv(path_1)
        df2 = pd.read_csv(path_2)
        df1['knight'] = df1['knight'].map({'Sith': 0, 'Jedi': 1})

        for column in df1.columns[:-1]:
            df1[column] = (df1[column] - df1[column].mean()) / df1[column].std()

        for column in df2.columns:
            df2[column] = (df2[column] - df2[column].mean()) / df2[column].std()

        print(df1)
        print(df2)

        df_jedi = df1[df1['knight'] == 1]
        df_sith = df1[df1['knight'] == 0]

        plt.figure(figsize=(12, 6))
        # Plot Sensitivity vs Hability for Jedi and Sith
        plt.scatter(df_jedi['Sensitivity'], df_jedi['Hability'], color='blue', label='Jedi', alpha=0.6)
        plt.scatter(df_sith['Sensitivity'], df_sith['Hability'], color='red', label='Sith', alpha=0.6)
        plt.title('Sensitivity vs Hability (with classification)')
        plt.xlabel('Sensitivity')
        plt.ylabel('Hability')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    standardize("../Train_knight.csv", "../Test_knight.csv")
