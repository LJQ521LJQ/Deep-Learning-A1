import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Args:
        file_path: str - Path to the CSV file
    Returns:
        X: numpy array of shape (n_samples, n_features) - Features
        y: numpy array of shape (n_samples,) - Labels
    """
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values  # Features (skipping the first column)
    y = data.iloc[:, 0].values   # Labels (first column)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        X: numpy array - Features
        y: numpy array - Labels
        test_size: float - Proportion of the dataset to include in the test split
        random_state: int - Random seed
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
