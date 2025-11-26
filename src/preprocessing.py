"""Preprocessing pipeline for Online Shoppers dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set (default 0.2)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


class OnlineShoppersPreprocessor:
    """
    Preprocessing pipeline for Online Shoppers dataset.

    Transformation steps:
    1. Identify numeric vs categorical features
    2. Label encode categorical features (Month, VisitorType, etc.)
    3. Convert boolean Weekend to integer
    4. StandardScale numeric features (z-score normalization)

    Attributes:
        label_encoders (dict): Fitted LabelEncoder for each categorical feature
        scaler (StandardScaler): Fitted scaler for numeric features
        numeric_cols (list): Names of numeric columns
        categorical_cols (list): Names of categorical columns
    """

    def __init__(self):
        """Initialize preprocessor with empty state."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_cols = None
        self.categorical_cols = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit encoders and scaler on training data, then transform.

        Args:
            X: Training features (DataFrame)
            y: Target (unused, kept for sklearn API compatibility)

        Returns:
            np.ndarray: Preprocessed feature matrix of shape (n_samples, 17)
        """
        X = X.copy()  # Avoid modifying original

        # Define feature types (based on dataset specification)
        self.categorical_cols = [
            "Month",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
            "VisitorType",
        ]
        self.numeric_cols = [
            col for col in X.columns if col not in self.categorical_cols + ["Weekend"]
        ]

        # Label encode categorical features
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        # Convert boolean Weekend to integer (0/1)
        if "Weekend" in X.columns:
            X["Weekend"] = X["Weekend"].astype(int)

        # Standard scale numeric features (z = (x - μ) / σ)
        X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        return X.values  # Return numpy array for sklearn models

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform test data using fitted encoders and scaler.

        IMPORTANT: Must be called after fit_transform() to use fitted parameters.

        Args:
            X: Test features (DataFrame)

        Returns:
            np.ndarray: Preprocessed feature matrix

        Raises:
            AttributeError: If called before fit_transform()
        """
        if not self.label_encoders:
            raise AttributeError("Must call fit_transform() before transform()")

        X = X.copy()

        # Apply fitted label encoders
        for col in self.categorical_cols:
            X[col] = self.label_encoders[col].transform(X[col])

        # Convert boolean
        if "Weekend" in X.columns:
            X["Weekend"] = X["Weekend"].astype(int)

        # Apply fitted scaler
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        return X.values
