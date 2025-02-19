import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score


def load_data(train_path, test_path):
    """
    Load training and test datasets from CSV files.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): Training dataset.
            - test_df (pd.DataFrame): Test dataset.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_data(df, mode, scaler):
    """
    Preprocess dataset by standardizing numerical features and, if training mode, split data.

    Args:
        df (pd.DataFrame): The dataset to be processed.
        mode (str): The mode of preprocessing. 
                    - "train" for training/validation split.
                    - "test" for processing test data.

    Returns:
        - If mode == "train":
            Tuple (np.ndarray, np.ndarray, pd.Series, pd.Series): 
            (features_train, features_val, targets_train, targets_val)
            where:
            - features_train: Training feature set.
            - features_val: Validation feature set.
            - targets_train: Training target labels.
            - targets_val: Validation target labels.
        
        - If mode == "test":
            np.ndarray: Standardized test feature set.

    Notes:
        - The function applies `StandardScaler` to normalize numerical features.
        - In training mode, it splits data into 80% training and 20% validation.
        - In test mode, it only standardizes features without splitting.
    """
    if mode == "train":
        features = df.iloc[:, :-1]  # Features
        targets = df.iloc[:, -1]   # Target

        scaler = StandardScaler()
        standardized = scaler.fit_transform(features)

        # Split dataset: 80% for training, 20% for validation
        features_train, features_val, targets_train, targets_val = train_test_split(standardized, targets, test_size=0.2, random_state=42)

        return features_train, features_val, targets_train, targets_val, scaler

    elif mode == "test":
        standardized = scaler.transform(df)
        return standardized


def train_model(train_set, targets):
    """
    Train a Random Forest classifier on the provided dataset.

    A Random Forest Classifier is an ensemble learning method
    that builds multiple decision trees and combines their predictions
    to improve accuracy and reduce overfitting.

    It is better than a single Decision Tree because
    it averages multiple trees' outputs, making it more robust,
    less prone to overfitting, and generally achieving higher predictive performance.

    Args:
        train_set (np.ndarray): Standardized training features.
        targets (pd.Series): Target labels.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100)  # random_state=42

    model.fit(train_set, targets)  # Train the model
    return model


def evaluate_model(model, features_val, targets_val):
    """
    Evaluate the trained model using the validation set.
    The F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.

    Args:
        model (RandomForestClassifier): The trained classifier.
        features_val (np.ndarray): Validation feature set.
        targets_val (pd.Series or np.ndarray): Actual target labels for validation.

    Returns:
        float: The weighted F1-score of the model.

    Prints:
        - Classification report including precision, recall, and F1-score.
        - The F1-score of the model on the validation set.
    """
    predictions = model.predict(features_val)
    f1 = f1_score(targets_val, predictions, average="weighted")

    print(f"F1-Score: {f1:.2f}")
    return f1


def display_tree(model, feature_names, tree_index=0):
    """
    Display a decision tree from the trained Random Forest.

    Args:
        model (RandomForestClassifier): The trained classifier.
        feature_names (list): List of feature names.
        tree_index (int, optional): Index of the tree to visualize from the Random Forest. Defaults to 0.

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))
    plot_tree(model.estimators_[tree_index], feature_names=feature_names, class_names=["Jedi", "Sith"], filled=True)
    plt.title(f"Decision Tree {tree_index} from Random Forest")
    plt.show()


def save_predictions(model, test_set, output_file="Tree.txt"):
    """
    Generate predictions from the trained model on the test dataset and save them to a file.

    Args:
        model (RandomForestClassifier): The trained classifier.
        test_set (np.ndarray): Standardized test features.
        output_file (str, optional): Output filename for predictions. Defaults to "Tree.txt".

    Returns:
        None
    """
    predictions = model.predict(test_set)
    np.savetxt(output_file, predictions, fmt="%s")
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    try:
        train_df, test_df = load_data("../Train_knight.csv", "../Test_knight.csv")

        features_train, features_val, targets_train, targets_val, scaler = preprocess_data(train_df, "train", 0)
        test_set = preprocess_data(test_df, "test", scaler)  # Test set has no labels

        model = train_model(features_train, targets_train)
        f1 = evaluate_model(model, features_val, targets_val)

        display_tree(model, train_df.columns[:-1])
        save_predictions(model, test_set)

    except Exception as e:
        print(f"Error: {e}")
