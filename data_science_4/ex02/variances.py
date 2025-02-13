import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def analyze_variance_with_pca(csv_path):
    """Standardize data, apply PCA, and compute cumulative variance."""
    try:
        df = pd.read_csv(csv_path)
        df = df.select_dtypes(include=[np.number])

        # Standardize the data (mean=0, variance=1)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df)

        # Creates a Principal Component Analysis (PCA) object.
        pca = PCA()
        # Finds the principal components of the standardized dataset.
        # Principal components are "lines" that groups maximum points to get "clusters", principal direction of points
        pca.fit(standardized_data)

        # Returns an array where each value represents
        # the percentage of variance explained ("spread" around principal components)
        # by a specific principal component.
        # Convert to percentage.
        variances = pca.explained_variance_ratio_ * 100

        # Calculate the running total of explained variance.
        # This helps determine how much total variance
        # is captured by keeping a certain number of components.
        #[{x% variance for 1 component}, {y% variance for 2 components}, etc]
        cumulative_variance = np.cumsum(variances)


        # Find the number of components to reach 90% variance

        # By looking at cumulative variance, you can figure out how many components
        # to keep in order to represent most of the important information
        # in your data without keeping unnecessary details.

        # If you’re working with a large dataset with many features,
        # PCA helps you reduce the number of dimensions (features) you need to work with,
        # and the cumulative variance tells you how much of
        # the data's variation you're keeping after reducing dimensions.

        # The cumulative variance helps you determine the most efficient way
        # to keep the essential information from the data (in fewer dimensions)
        # while discarding the less important details.
        num_components = np.argmax(cumulative_variance >= 90) + 1

        print("Variances (Percentage):")
        print(variances)
        print("\nCumulative Variances (Percentage):")
        print(cumulative_variance)
        print(f"\nNumber of components to reach 90% variance: {num_components}")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, linestyle='-')
        plt.xlabel("Number of Components")
        plt.ylabel("Explained Variance (%)")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_variance_with_pca("../Train_knight.csv")
