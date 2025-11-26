"""Main pipeline for e-commerce purchase intention classification."""

import os

from src.data_loader import get_dataset_info, load_online_shoppers
from src.evaluation import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    print_classification_report,
)
from src.models import train_mlp, train_random_forest, train_svm
from src.preprocessing import OnlineShoppersPreprocessor, split_data


def main():
    """Execute complete ML pipeline."""

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 80)
    print("E-COMMERCE PURCHASE INTENTION PREDICTION PIPELINE")
    print("=" * 80)

    # 1. Load data
    print("\n[1/7] Loading UCI dataset (ID: 468)...")
    X, y = load_online_shoppers()
    get_dataset_info(X, y)

    # 2. Split data
    print("\n[2/7] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 3. Preprocessing
    print("\n[3/7] Preprocessing features...")
    preprocessor = OnlineShoppersPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed shape: {X_train_proc.shape}")
    print(f"Feature type: {X_train_proc.dtype}")

    # 4. Train MLP
    print("\n[4/7] Training MLP (GridSearchCV, ~5-10 minutes)...")
    model_mlp = train_mlp(X_train_proc, y_train)

    # 5. Train SVM
    print("\n[5/7] Training SVM (GridSearchCV, ~5-10 minutes)...")
    model_svm = train_svm(X_train_proc, y_train)

    # 6. Train Random Forest
    print("\n[6/7] Training Random Forest (GridSearchCV, ~5-10 minutes)...")
    model_rf = train_random_forest(X_train_proc, y_train)

    # 7. Evaluate all models
    print("\n[7/7] Evaluating models on test set...")
    results = []

    # MLP evaluation
    metrics_mlp = evaluate_model(model_mlp, X_test_proc, y_test, "MLP")
    results.append(metrics_mlp)
    print_classification_report(y_test, model_mlp.predict(X_test_proc), "MLP")

    # SVM evaluation
    metrics_svm = evaluate_model(model_svm, X_test_proc, y_test, "SVM")
    results.append(metrics_svm)
    print_classification_report(y_test, model_svm.predict(X_test_proc), "SVM")

    # Random Forest evaluation
    metrics_rf = evaluate_model(model_rf, X_test_proc, y_test, "Random Forest")
    results.append(metrics_rf)
    print_classification_report(y_test, model_rf.predict(X_test_proc), "Random Forest")

    # 8. Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        y_test,
        model_mlp.predict(X_test_proc),
        "MLP",
        "results/confusion_matrix_mlp.png",
    )
    plot_confusion_matrix(
        y_test,
        model_svm.predict(X_test_proc),
        "SVM",
        "results/confusion_matrix_svm.png",
    )
    plot_confusion_matrix(
        y_test,
        model_rf.predict(X_test_proc),
        "Random Forest",
        "results/confusion_matrix_random_forest.png",
    )

    models = {"MLP": model_mlp, "SVM": model_svm, "Random Forest": model_rf}
    plot_roc_curve(models, X_test_proc, y_test, "results/roc_curves.png")

    # 9. Save comparison
    df_results = compare_models(results)
    df_results.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved comparison to results/model_comparison.csv")

    # 10. Display final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(df_results.to_string(index=False))

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("Results saved to 'results/' directory:")
    print("  - model_comparison.csv")
    print("  - confusion_matrix_mlp.png")
    print("  - confusion_matrix_svm.png")
    print("  - confusion_matrix_random_forest.png")
    print("  - roc_curves.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
