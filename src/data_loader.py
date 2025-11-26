"""UCI Online Shoppers Dataset loading utilities."""

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_online_shoppers() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load UCI Online Shoppers Purchasing Intention Dataset (ID: 468).

    Returns:
        tuple: (X, y) where:
            - X: DataFrame of shape (12330, 17) with features
            - y: Series of shape (12330,) with binary target (True/False)

    Raises:
        Exception: If dataset download fails
    """
    try:
        dataset = fetch_ucirepo(id=468)
        X = dataset.data.features
        y = dataset.data.targets.squeeze()  # Convert single-column DataFrame to Series

        # Validate data
        assert X.shape == (12330, 17), f"Expected shape (12330, 17), got {X.shape}"
        assert len(y) == 12330, f"Expected 12330 targets, got {len(y)}"

        return X, y
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def get_dataset_info(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Print comprehensive dataset statistics for verification.

    Args:
        X: Feature DataFrame
        y: Target Series

    Displays:
        - Shape and column types
        - Missing values count
        - Class distribution (imbalance ratio)
        - Basic descriptive statistics
    """
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    print(f"\nFeatures Shape: {X.shape}")
    print(f"Target Shape: {y.shape}")

    print("\nFeature Types:")
    print(X.dtypes)

    print("\nMissing Values:")
    missing = X.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    print("\nTarget Distribution (Revenue):")
    print(y.value_counts())
    print("\nProportions:")
    print(y.value_counts(normalize=True).round(4))

    print(f"\nClass Imbalance Ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")

    print("\nNumeric Features Summary:")
    print(X.describe())

    print("=" * 80)
