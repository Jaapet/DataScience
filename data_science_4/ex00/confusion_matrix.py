import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_labels(path):
    """Load labels from a text file into a list."""
    with open(path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def compute(y_true, y_pred, classes):
    """
    Compute the confusion matrix.

    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        classes (list): Sorted list of unique class labels.

    Returns:
        np.ndarray: A confusion matrix of shape (num_classes, num_classes),
                    where element (i, j) represents the number of times
                    class i was predicted as class j.
    """

    # Initialize a square confusion matrix with zeros.
    # Each row represents the true class, and each column represents the predicted class.
    # zip() combines multiple iterables into a single iterable of tuples.
    # Each tuple contains one element from each iterable at the same index.
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    # Iterate over the pairs of true labels and predicted labels
    for true_label, pred_label in zip(y_true, y_pred):
        # Find the indexes of labels
        true_index = classes.index(true_label)
        pred_index = classes.index(pred_label)

        # Increment the corresponding cell in the confusion matrix
        # The row represents the true class,
        # and the column represents the predicted class.
        matrix[true_index, pred_index] += 1

    return matrix

def metrics(matrix, classes):
    """
    Compute classification metrics: precision, recall, F1-score, total samples per class, and accuracy.

    Args:
        matrix (np.ndarray): The confusion matrix.
        classes (list): Sorted list of unique class labels.

    Returns:
        tuple:
            - dict: A dictionary where keys are class labels and values are metric dictionaries.
            - float: Overall accuracy rounded to 2 decimal places.
    """
    metrics = {}

    # Compute the total number of correctly classified samples (sum of diagonal elements)
    # In a confusion matrix, these diagonal elements represent the correctly classified samples (True Positives for each class).
    total_correct = np.trace(matrix)

    total_samples = np.sum(matrix)
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    for i in range(len(classes)):
        # True Positives (TP): The number of correctly predicted samples for class i
        TP = matrix[i, i]

        # False Positives (FP): The number of times class i was incorrectly predicted
        # Sum of column i (all predicted as class i) minus the true positives
        FP = sum(matrix[:, i]) - TP

        # False Negatives (FN): The number of times class i was not correctly predicted
        # Sum of row i (all true class i samples) minus the true positives
        FN = sum(matrix[i, :]) - TP

        total = sum(matrix[i, :])

        # Precision: proportion of correct predictions for this class
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # Recall: proportion of actual instances correctly predicted
        # Recall measures how well the model identifies actual positive cases.
        # A high recall means the model rarely misses positives.
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # F1-score: Harmonic mean of precision and recall
        # The F1-score balances Precision and Recall into a single metric.
        # It is the harmonic mean of Precision and Recall, giving a good measure when classes are imbalanced.
        # A high F1-score means both Precision and Recall are high. If one is low, F1-score will drop.
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[classes[i]] = {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1-score": round(f1_score, 2),
            "Total": total
        }

    return metrics, round(accuracy, 2)

def show_plot(matrix, classes):
    """
    Display the confusion matrix using a heatmap.

    Args:
        matrix (np.ndarray): The confusion matrix.
        classes (list): Sorted list of unique class labels.

    Returns:
        None
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    try :
        true_labels = load_labels("../truth.txt")
        pred_labels = load_labels("../predictions.txt")

        classes = sorted(set(true_labels + pred_labels))

        matrix = compute(true_labels, pred_labels, classes)

        print("Confusion Matrix:")
        print(matrix)

        metrics, accuracy = metrics(matrix, classes)
        print("\nMetrics:")
        for cls, values in metrics.items():
            print(f"{cls}: Precision={values['Precision']}, Recall={values['Recall']}, F1-score={values['F1-score']}, Total={values['Total']}")

        print(f"\nAccuracy: {accuracy}")

        show_plot(matrix, classes)

    except Exception as e:
        print(f"Error: {e}")
