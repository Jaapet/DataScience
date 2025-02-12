import pandas as pd
import matplotlib.pyplot as plt

def normalize(path_1: str, path_2: str):
    try:
        df1 = pd.read_csv(path_1)
        df2 = pd.read_csv(path_2)
        df1['knight'] = df1['knight'].map({'Sith': 0, 'Jedi': 1})

        for column in df1.columns[:-1]:
            df1[column] = (df1[column] - df1[column].min()) / (df1[column].max() - df1[column].min())

        for column in df2.columns:
            df2[column] = (df2[column] - df2[column].min()) / (df2[column].max() - df2[column].min())

        print(df1)
        print(df2)

        df_jedi = df1[df1['knight'] == 1]
        df_sith = df1[df1['knight'] == 0]

        plt.figure(figsize=(12, 6))

        plt.scatter(df_jedi['Push'], df_jedi['Deflection'], color='blue', label='Jedi', alpha=0.6)
        plt.scatter(df_sith['Push'], df_sith['Deflection'], color='red', label='Sith', alpha=0.6)
        plt.title('Push vs Deflection (with classification)')
        plt.xlabel('Push')
        plt.ylabel('Deflection')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    normalize("../Train_knight.csv", "../Test_knight.csv")
