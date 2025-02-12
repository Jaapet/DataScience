import pandas as pd
import matplotlib.pyplot as plt


def plot_graphs(path_1: str, path_2: str):
    try:
        df_1 = pd.read_csv(path_1)
        df_1['knight'] = df_1['knight'].map({'Sith': 0, 'Jedi': 1})

        df_jedi_1 = df_1[df_1['knight'] == 1]
        df_sith_1 = df_1[df_1['knight'] == 0]

        df_2 = pd.read_csv(path_2)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plot 1: Sensitivity vs Hability (with classification)
        axes[0, 0].scatter(df_jedi_1['Sensitivity'], df_jedi_1['Hability'], color='blue', label='Jedi', alpha=0.6)
        axes[0, 0].scatter(df_sith_1['Sensitivity'], df_sith_1['Hability'], color='red', label='Sith', alpha=0.6)
        axes[0, 0].set_title('Sensitivity vs Hability (with classification)')
        axes[0, 0].set_xlabel('Sensitivity')
        axes[0, 0].set_ylabel('Hability')
        axes[0, 0].legend()

        # Plot 2: Push vs Deflection (with classification)
        axes[0, 1].scatter(df_jedi_1['Push'], df_jedi_1['Deflection'], color='blue', label='Jedi', alpha=0.6)
        axes[0, 1].scatter(df_sith_1['Push'], df_sith_1['Deflection'], color='red', label='Sith', alpha=0.6)
        axes[0, 1].set_title('Push vs Deflection (with classification)')
        axes[0, 1].set_xlabel('Push')
        axes[0, 1].set_ylabel('Deflection')
        axes[0, 1].legend()

        # Plot 3: Sensitivity vs Hability (without classification)
        axes[1, 0].scatter(df_2['Sensitivity'], df_2['Hability'], color='green', alpha=0.6)
        axes[1, 0].set_title('Sensitivity vs Hability (without classification)')
        axes[1, 0].set_xlabel('Sensitivity')
        axes[1, 0].set_ylabel('Hability')

        # Plot 4: Push vs Deflection (without classification)
        axes[1, 1].scatter(df_2['Push'], df_2['Deflection'], color='green', alpha=0.6)
        axes[1, 1].set_title('Push vs Deflection (without classification)')
        axes[1, 1].set_xlabel('Push')
        axes[1, 1].set_ylabel('Deflection')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    plot_graphs("../Train_knight.csv", "../Test_knight.csv")
