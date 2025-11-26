"""Model evaluation metrics and visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """
    Evaluate model and return comprehensive metrics.

    Args:
        model: Trained sklearn model with predict() and predict_proba()
        X_test: Test features
        y_test: Test labels
        model_name: Name for display in results

    Returns:
        dict: Metrics dictionary with keys:
            - Model, Accuracy, Precision, Recall, F1-Score, AUC-ROC
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
    }

    return metrics


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: str
) -> None:
    """
    Generate and save confusion matrix heatmap.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Model name for title
        save_path: Full path to save PNG file
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,  # Show numbers in cells
        fmt="d",  # Integer format
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Prevent memory leak

    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(
    models_dict: dict, X_test: np.ndarray, y_test: np.ndarray, save_path: str
) -> None:
    """
    Plot ROC curves for multiple models on same figure.

    Args:
        models_dict: Dictionary mapping model names to fitted model objects
                    Example: {'MLP': model_mlp, 'SVM': model_svm}
        X_test: Test features
        y_test: Test labels
        save_path: Full path to save PNG file
    """
    plt.figure(figsize=(10, 8))

    # Plot each model's ROC curve
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

    # Diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved ROC curves to {save_path}")


def compare_models(results_list: list) -> pd.DataFrame:
    """
    Create comparison DataFrame from results.

    Args:
        results_list: List of metric dictionaries from evaluate_model()

    Returns:
        pd.DataFrame: Comparison table sorted by F1-Score (descending)
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values("F1-Score", ascending=False)

    # Round numeric columns for readability
    numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def print_classification_report(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Print detailed classification report to console.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Model name for display
    """
    print(f"\n{'=' * 80}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print(f"{'=' * 80}")
    print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"], digits=4))
