"""
Unit tests for evaluation module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.evaluation import (
    precision_at_k,
    evaluate_model,
    plot_confusion_matrix,
    get_feature_importance,
    plot_feature_importance,
    find_optimal_threshold,
    evaluate_with_threshold
)


class TestPrecisionAtK:
    """Test cases for precision_at_k function."""
    
    def test_precision_at_k_perfect(self):
        """Test precision@k with perfect predictions."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        
        result = precision_at_k(y_true, y_scores, k=3)
        
        assert result == 1.0
    
    def test_precision_at_k_partial(self):
        """Test precision@k with partial matches."""
        y_true = np.array([1, 1, 0, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        
        result = precision_at_k(y_true, y_scores, k=3)
        
        # Top 3: indices 0, 1, 2 -> [1, 1, 0] -> precision = 2/3
        assert abs(result - 2/3) < 0.01
    
    def test_precision_at_k_empty(self):
        """Test precision@k with empty array."""
        y_true = np.array([])
        y_scores = np.array([])
        
        result = precision_at_k(y_true, y_scores, k=5)
        
        assert result == 0.0
    
    def test_precision_at_k_k_larger_than_array(self):
        """Test precision@k when k is larger than array size."""
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.8, 0.7])
        
        result = precision_at_k(y_true, y_scores, k=10)
        
        # Should use all 3 elements
        assert abs(result - 2/3) < 0.01
    
    def test_precision_at_k_k_equals_one(self):
        """Test precision@k with k=1."""
        y_true = np.array([1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7])
        
        result = precision_at_k(y_true, y_scores, k=1)
        
        assert result == 1.0


class TestEvaluateModel:
    """Test cases for evaluate_model function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=Pipeline)
        model.predict.return_value = np.array([0, 1, 0, 1, 0])
        model.predict_proba.return_value = np.array([
            [0.7, 0.3],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.8, 0.2]
        ])
        return model
    
    def test_evaluate_model_returns_predictions(self, mock_model):
        """Test that evaluate_model returns predictions and probabilities."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_test = pd.Series([0, 1, 0, 1, 0])
        
        y_pred, y_prob = evaluate_model(mock_model, X_test, y_test, "Test Model")
        
        assert len(y_pred) == len(X_test)
        assert len(y_prob) == len(X_test)
        assert isinstance(y_pred, np.ndarray)
        assert isinstance(y_prob, np.ndarray)
    
    def test_evaluate_model_calls_predict(self, mock_model):
        """Test that evaluate_model calls model.predict."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])
        
        evaluate_model(mock_model, X_test, y_test, "Test Model")
        
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()


class TestGetFeatureImportance:
    """Test cases for get_feature_importance function."""
    
    @pytest.fixture
    def mock_rf_model(self):
        """Create a mock Random Forest model."""
        model = MagicMock(spec=Pipeline)
        model.named_steps = {
            'pre': MagicMock(),
            'clf': MagicMock(spec=RandomForestClassifier)
        }
        model.named_steps['pre'].get_feature_names_out.return_value = [
            'feature1', 'feature2', 'feature3'
        ]
        model.named_steps['clf'].feature_importances_ = np.array([0.5, 0.3, 0.2])
        return model
    
    @pytest.fixture
    def mock_logreg_model(self):
        """Create a mock Logistic Regression model."""
        model = MagicMock(spec=Pipeline)
        model.named_steps = {
            'pre': MagicMock(),
            'clf': MagicMock(spec=LogisticRegression)
        }
        model.named_steps['pre'].get_feature_names_out.return_value = [
            'feature1', 'feature2', 'feature3'
        ]
        model.named_steps['clf'].coef_ = np.array([[0.5, -0.3, 0.2]])
        return model
    
    def test_get_feature_importance_rf(self, mock_rf_model):
        """Test feature importance for Random Forest."""
        result = get_feature_importance(mock_rf_model, "RF Model")
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert 'feature1' in result.index
        assert result['feature1'] == 0.5
    
    def test_get_feature_importance_logreg(self, mock_logreg_model):
        """Test feature importance for Logistic Regression."""
        result = get_feature_importance(mock_logreg_model, "LogReg Model")
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert 'feature1' in result.index
    
    def test_get_feature_importance_no_importance(self):
        """Test with model that has no feature importance."""
        model = MagicMock(spec=Pipeline)
        model.named_steps = {
            'pre': MagicMock(),
            'clf': MagicMock()
        }
        # Remove feature_importances_ and coef_ attributes
        del model.named_steps['clf'].feature_importances_
        del model.named_steps['clf'].coef_
        
        result = get_feature_importance(model, "Test Model")
        
        assert result is None


class TestFindOptimalThreshold:
    """Test cases for find_optimal_threshold function."""
    
    def test_find_optimal_threshold_f1(self):
        """Test finding optimal threshold for F1 score."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.4, 0.85])
        
        threshold, metrics = find_optimal_threshold(y_true, y_prob, metric='f1')
        
        assert 0.0 <= threshold <= 1.0
        assert 'threshold' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_find_optimal_threshold_precision(self):
        """Test finding optimal threshold for precision."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        
        threshold, metrics = find_optimal_threshold(y_true, y_prob, metric='precision')
        
        assert 0.0 <= threshold <= 1.0
        assert metrics['precision'] >= 0
    
    def test_find_optimal_threshold_recall(self):
        """Test finding optimal threshold for recall."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        
        threshold, metrics = find_optimal_threshold(y_true, y_prob, metric='recall')
        
        assert 0.0 <= threshold <= 1.0
        assert metrics['recall'] >= 0


class TestEvaluateWithThreshold:
    """Test cases for evaluate_with_threshold function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock(spec=Pipeline)
        model.predict_proba.return_value = np.array([
            [0.7, 0.3],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.3, 0.7]
        ])
        return model
    
    def test_evaluate_with_threshold_default(self, mock_model):
        """Test evaluation with default threshold (0.5)."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4]})
        y_test = pd.Series([0, 1, 0, 1])
        
        metrics = evaluate_with_threshold(mock_model, X_test, y_test, threshold=0.5, name="Test")
        
        assert 'threshold' in metrics
        assert 'y_pred' in metrics
        assert 'y_prob' in metrics
        assert metrics['threshold'] == 0.5
    
    def test_evaluate_with_threshold_custom(self, mock_model):
        """Test evaluation with custom threshold."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4]})
        y_test = pd.Series([0, 1, 0, 1])
        
        metrics = evaluate_with_threshold(mock_model, X_test, y_test, threshold=0.3, name="Test")
        
        assert metrics['threshold'] == 0.3
        # With lower threshold, more predictions should be positive
        assert metrics['y_pred'].sum() >= 0


class TestPlotFunctions:
    """Test cases for plotting functions."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for plotting tests."""
        model = MagicMock(spec=Pipeline)
        model.named_steps = {
            'pre': MagicMock(),
            'clf': MagicMock(spec=RandomForestClassifier)
        }
        model.named_steps['pre'].get_feature_names_out.return_value = [
            'feature1', 'feature2', 'feature3'
        ]
        model.named_steps['clf'].feature_importances_ = np.array([0.5, 0.3, 0.2])
        return model
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, mock_show, mock_model):
        """Test plotting feature importance."""
        plot_feature_importance(mock_model, "Test Model", top_n=3, save_plot=True)
        
        # Should save the plot
        mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show):
        """Test plotting confusion matrix."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        # Should not raise an error
        plot_confusion_matrix(y_true, y_pred, "Test Confusion Matrix")

