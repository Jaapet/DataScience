import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score
from sklearn.preprocessing import LabelEncoder


def load_data(train_file, test_file):
    """
    Load the training and testing data from CSV files.

    Args:
        train_file (str): The path to the training data CSV file.
        test_file (str): The path to the testing data CSV file.

    Returns:
        tuple: A tuple containing:
            - features_train (pd.DataFrame): The feature columns from the training data.
            - targets_train (pd.Series): The target labels from the training data.
            - test_data (pd.DataFrame): The testing data (without labels).
    """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    features_train = train_data.iloc[:, :-1]
    targets_train = train_data.iloc[:, -1]
    
    return features_train, targets_train, test_data


def evaluate_knn(features_train, targets_train, X_val, y_val):
    """
    Evaluate the KNN model to find the optimal number of neighbors (k) based on the F1-score.
    The value k represents the number of nearest neighbors the algorithm considers
    when making a classification decision.
    So, for each k in this range, the model will be trained, evaluated,
    and the corresponding F1-score will be calculated.

    Args:
        features_train (pd.DataFrame): The feature columns from the training data.
        targets_train (pd.Series): The target labels from the training data.
        X_val (pd.DataFrame): The feature columns from the validation data.
        y_val (pd.Series): The target labels from the validation data.

    Returns:
        tuple: A tuple containing:
            - best_k (int): The optimal number of neighbors (k) for the KNN model.
            - max_f1_score (float): The maximum F1-score achieved with the optimal k.
    """
    f1_scores = []
    k_values = range(1, 21)  # We will check k values from 1 to 20

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(features_train, targets_train)
        predictions = knn.predict(X_val)
        f1 = f1_score(y_val, predictions, average='weighted')
        f1_scores.append(f1)

    plt.plot(k_values, f1_scores, marker='o')
    plt.title('F1-Score vs. k value')
    plt.xlabel('k value')
    plt.ylabel('F1-Score')
    plt.show()

    # Find the optimal k that gives a minimum F1-score of 92%
    best_k = k_values[np.argmax(f1_scores)]
    return best_k, max(f1_scores)


def knn_classification(features_train, targets_train, features_test, best_k):
    """
    Train a KNN model with the optimal number of neighbors (k) and make predictions on the test data.

    Args:
        features_train (pd.DataFrame): The feature columns from the training data.
        targets_train (pd.Series): The target labels from the training data.
        features_test (pd.DataFrame): The feature columns from the test data.
        best_k (int): The optimal number of neighbors (k) for the KNN model.

    Returns:
        np.ndarray: An array of predicted labels for the test data.
    """
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(features_train, targets_train)

    predictions = knn.predict(features_test)

    return predictions


def save_predictions(predictions, output_file):
    """
    Save the predictions to a text file, with one prediction per line.

    Args:
        predictions (np.ndarray): An array of predicted labels.
        output_file (str): The path to the output file where predictions will be saved.
    """
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(pred + '\n')


if __name__ == "__main__":
    try:
        features_train, targets_train, features_test = load_data("../Train_knight.csv", "../Test_knight.csv")

        le = LabelEncoder()
        targets_train = le.fit_transform(targets_train)

        features_train, X_val, targets_train, y_val = train_test_split(features_train, targets_train, test_size=0.2)  # random_state=42

        best_k, f1_score_val = evaluate_knn(features_train, targets_train, X_val, y_val)

        predictions = knn_classification(features_train, targets_train, features_test, best_k)

        predictions = le.inverse_transform(predictions)
        save_predictions(predictions, "KNN.txt")

    except Exception as e:
        print(f"Error: {e}")
