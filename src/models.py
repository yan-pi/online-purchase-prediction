"""Model training with GridSearchCV hyperparameter tuning."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def train_mlp(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train Multi-Layer Perceptron with GridSearchCV.

    Hyperparameter grid:
    - hidden_layer_sizes: [(100,50), (100,), (50,50)]
    - alpha (L2 regularization): [0.0001, 0.001]
    - activation: relu (fixed)
    - solver: adam (fixed)
    - learning_rate: adaptive (fixed)

    Args:
        X_train: Training features (n_samples, 17)
        y_train: Training labels (n_samples,)

    Returns:
        MLPClassifier: Best estimator from grid search
    """
    param_grid = {
        "hidden_layer_sizes": [(100, 50), (100,), (50, 50)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["adaptive"],
        "max_iter": [500],
    }

    mlp = MLPClassifier(
        early_stopping=True,  # Prevent overfitting
        validation_fraction=0.1,  # 10% for validation
        random_state=42,
    )

    grid_search = GridSearchCV(
        mlp,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",  # Better than accuracy for imbalanced data
        n_jobs=-1,  # Use all CPU cores
        verbose=2,  # Show progress
    )

    print("Starting MLP GridSearchCV (this may take 5-10 minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest MLP parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Train Support Vector Machine with GridSearchCV.

    Hyperparameter grid:
    - C (regularization): [0.1, 1, 10]
    - kernel: ['rbf', 'linear']
    - gamma: ['scale', 'auto']

    Args:
        X_train: Training features (n_samples, 17)
        y_train: Training labels (n_samples,)

    Returns:
        SVC: Best estimator from grid search
    """
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    }

    svm = SVC(
        class_weight="balanced",  # Handle class imbalance
        probability=True,  # Required for AUC-ROC calculation
        random_state=42,
        cache_size=500,  # Speed up training (MB)
    )

    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        verbose=2,
    )

    print("Starting SVM GridSearchCV (this may take 5-10 minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest SVM parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest with GridSearchCV.

    Hyperparameter grid:
    - n_estimators: [50, 100, 200]
    - max_depth: [10, 20, None]
    - min_samples_split: [2, 5]
    - class_weight: balanced (fixed)

    Args:
        X_train: Training features (n_samples, 17)
        y_train: Training labels (n_samples,)

    Returns:
        RandomForestClassifier: Best estimator from grid search
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        verbose=2,
    )

    print("Starting Random Forest GridSearchCV (this may take 5-10 minutes)...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest RF parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_
