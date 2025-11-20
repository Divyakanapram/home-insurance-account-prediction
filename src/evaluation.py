"""
Model evaluation and explainability module.

Provides functions for model evaluation, threshold optimization, and explainability
using SHAP values and feature importance plots.
"""

from typing import Tuple, Optional, Dict, Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib

# Force a headless-friendly backend before pyplot import
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: SHAP not available. Install with: pip install shap")

# Output directory for explainability plots
EXPLAINABILITY_OUTPUT_DIR = Path("output/explainability_plots")
EXPLAINABILITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Calculate precision at k (top k predictions).
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Predicted probabilities or scores
    k : int
        Number of top predictions to consider
    
    Returns:
    --------
    float : Precision at k
    """
    if len(y_true) == 0:
        return 0.0
    k_eff = min(k, len(y_true))
    idx = np.argsort(y_scores)[::-1][:k_eff]
    return np.mean(np.array(y_true)[idx])


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model",
    additional_thresholds: Optional[Sequence[float]] = (0.4,),
) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    expected_len = len(y_test)
    if len(y_pred) != expected_len:
        warnings.warn(
            f"Predictions length ({len(y_pred)}) does not match y_test length ({expected_len}). "
            "Truncating to match.",
            RuntimeWarning,
        )
        y_pred = y_pred[:expected_len]

    if len(y_prob) != expected_len:
        warnings.warn(
            f"Predicted probabilities length ({len(y_prob)}) does not match y_test length ({expected_len}). "
            "Truncating to match.",
            RuntimeWarning,
        )
        y_prob = y_prob[:expected_len]


    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    
    # Precision@K metrics (as per notebook)
    print("\nPrecision@K metrics:")
    for k in [50, 100, 200]:
        k_eff = min(k, len(y_test))
        prec_at_k = precision_at_k(y_test.values, y_prob, k_eff)
        print(f"  Precision@{k_eff}: {prec_at_k:.4f}")


    plot_conf_matrix(y_test, y_pred, name)
    _report_additional_thresholds(y_test, y_prob, y_pred, additional_thresholds)
    
    return y_pred, y_prob


def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    try:
        plt.figure(figsize=(6, 5))
    except Exception as exc:  # Handle headless environments lacking Tk
        warnings.warn(
            f"Falling back to Agg backend for plotting due to: {exc}",
            RuntimeWarning,
        )
        matplotlib.use("Agg", force=True)
        plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(title)
    plt.show()


# Backwards compatibility alias for older imports/tests
plot_confusion_matrix = plot_conf_matrix


def explain_model_shap(model, X_test, model_name="Model", model_type="rf", save_plots=True):
    """
    Generate SHAP explanations for a model.
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained model pipeline with 'pre' (preprocessor) and 'clf' (classifier) steps
    X_test : pd.DataFrame
        Test features (raw, before preprocessing)
    model_name : str
        Name of the model for display
    model_type : str
        Type of model: 'rf' for Random Forest, 'logreg' for Logistic Regression
    save_plots : bool
        Whether to save plots to file
    
    Returns:
    --------
    shap_values : array or list
        SHAP values
    feature_names : list
        Feature names after preprocessing
    """
    if not SHAP_AVAILABLE:
        print(f"SHAP not available. Skipping explainability for {model_name}.")
        print("Install SHAP with: pip install shap")
        return None, None
    
    try:
        # Get preprocessed test data
        X_test_pre = model.named_steps['pre'].transform(X_test)
        
        # Get feature names from preprocessor
        feature_names = model.named_steps['pre'].get_feature_names_out()
        
        # Convert to DataFrame with feature names
        X_test_pre_df = pd.DataFrame(X_test_pre, columns=feature_names, index=X_test.index)
        
        if model_type.lower() == 'rf':
            # Random Forest: Use TreeExplainer (as per notebook Cell 69)
            print(f"\nGenerating SHAP explanations for {model_name} (Random Forest)...")
            explainer = shap.TreeExplainer(model.named_steps['clf'])
            shap_values = explainer.shap_values(X_test_pre_df)
            
            # For binary classification, shap_values might be a list [class_0, class_1]
            # Use class_1 (positive class) for explanation
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Summary plot
            print("Creating SHAP summary plot...")
            shap.summary_plot(shap_values, features=X_test_pre_df, feature_names=feature_names, show=False)
            plt.title(f"{model_name} - SHAP Summary Plot")
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                filename = f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
                filepath = EXPLAINABILITY_OUTPUT_DIR / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"✓ Saved SHAP plot: {filepath}")
            plt.show()
            
        elif model_type.lower() in ['logreg', 'logistic']:
            # Logistic Regression: Use Explainer (as per notebook Cell 68)
            print(f"\nGenerating SHAP explanations for {model_name} (Logistic Regression)...")
            # Get the trained classifier
            logreg_clf = model.named_steps['clf']
            explainer = shap.Explainer(logreg_clf, X_test_pre)
            shap_values = explainer(X_test_pre)
            
            print(f"SHAP shape: {shap_values.shape}")
            print(f"X_test_preprocessed shape: {X_test_pre.shape}")
            
            # Summary plot for linear model
            print("Creating SHAP summary plot...")
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"{model_name} - SHAP Summary Plot")
            plt.tight_layout()
            
            # Save plot
            if save_plots:
                filename = f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
                filepath = EXPLAINABILITY_OUTPUT_DIR / filename
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"✓ Saved SHAP plot: {filepath}")
            plt.show()
        else:
            print(f"Unknown model type: {model_type}. Supported: 'rf', 'logreg'")
            return None, None
        
        return shap_values, feature_names
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_feature_importance(model, model_name="Model"):
    """
    Get feature importance/coefficients from trained models.
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained model pipeline
    model_name : str
        Name of the model
    
    Returns:
    --------
    feature_importance : pd.Series
        Feature importance/coefficients
    """
    try:
        # Get feature names
        feature_names = model.named_steps['pre'].get_feature_names_out()
        clf = model.named_steps['clf']
        
        if hasattr(clf, 'feature_importances_'):
            # Random Forest
            importances = clf.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        elif hasattr(clf, 'coef_'):
            # Logistic Regression
            coefs = clf.coef_[0]
            feat_imp = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False)
        else:
            print(f"Model {model_name} does not have feature importance or coefficients")
            return None
        
        return feat_imp
        
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return None


def plot_feature_importance(model, model_name="Model", top_n=20, save_plot=True):
    """
    Plot top N feature importances.
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained model pipeline
    model_name : str
        Name of the model
    top_n : int
        Number of top features to plot
    save_plot : bool
        Whether to save the plot to file
    """
    feat_imp = get_feature_importance(model, model_name)
    if feat_imp is None:
        return
    
    top_features = feat_imp.head(top_n)
    
    plt.figure(figsize=(8, 6))
    top_features.plot(kind='barh')
    plt.title(f'Top {top_n} {model_name} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        filepath = EXPLAINABILITY_OUTPUT_DIR / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance plot: {filepath}")
    
    plt.show()


def _report_additional_thresholds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    baseline_pred: np.ndarray,
    thresholds: Optional[Sequence[float]],
) -> None:
    """
    Print evaluation metrics for additional probability thresholds.

    Parameters
    ----------
    y_true : pd.Series
        Ground-truth labels.
    y_prob : np.ndarray
        Predicted probabilities for positive class.
    baseline_pred : np.ndarray
        Predictions generated by `model.predict` (assumed 0.5 threshold).
    thresholds : Sequence[float] | None
        Additional thresholds to evaluate (duplicates ignored).
    """
    if not thresholds:
        return

    # Preserve order but remove duplicates
    unique_thresholds = []
    for thr in thresholds:
        if thr is None:
            continue
        if thr not in unique_thresholds:
            unique_thresholds.append(thr)

    if not unique_thresholds:
        return

    print("\nAdditional threshold evaluations:")
    for thr in unique_thresholds:
        if not (0.0 <= thr <= 1.0):
            print(f"  - Skipping invalid threshold {thr}. Must be between 0 and 1.")
            continue

        if np.isclose(thr, 0.5):
            y_pred_thr = baseline_pred
            note = " (matches default scikit-learn threshold)"
        else:
            y_pred_thr = (y_prob >= thr).astype(int)
            note = ""

        print(f"\n--- Threshold = {thr:.2f}{note} ---")
        print(classification_report(y_true, y_pred_thr))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred_thr))


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """
    Find optimal threshold for binary classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall', or 'roc_auc'
    
    Returns:
    --------
    float : Optimal threshold
    dict : Metrics at optimal threshold
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_thresh)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_thresh)
        else:
            score = f1_score(y_true, y_pred_thresh)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1_score(y_true, y_pred_thresh),
                'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh)
            }
    
    return best_threshold, best_metrics


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5, name="Model"):
    """
    Evaluate model with custom threshold.
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : array-like
        True labels
    threshold : float
        Classification threshold
    name : str
        Model name
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    
    print(f"=== {name} (Threshold={threshold}) ===")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    
    metrics = {
        'threshold': threshold,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    return metrics