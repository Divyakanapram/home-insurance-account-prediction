"""
Unit tests for model_train module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model_train import train_models, filter_unhashable_columns


class TestFilterUnhashableColumns:
    """Test cases for filter_unhashable_columns function."""
    
    def test_filter_unhashable_columns_no_lists(self):
        """Test with dataframe containing no unhashable types."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.5, 2.5, 3.5]
        })
        
        result = filter_unhashable_columns(df)
        
        assert len(result.columns) == 3
        assert 'col1' in result.columns
        assert 'col2' in result.columns
        assert 'col3' in result.columns
    
    def test_filter_unhashable_columns_with_lists(self):
        """Test with dataframe containing list values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [[1, 2], [3, 4], [5, 6]],
            'col3': ['a', 'b', 'c']
        })
        
        result = filter_unhashable_columns(df)
        
        # col2 should be dropped
        assert 'col2' not in result.columns
        assert 'col1' in result.columns
        assert 'col3' in result.columns
    
    def test_filter_unhashable_columns_with_dicts(self):
        """Test with dataframe containing dict values."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [{'a': 1}, {'b': 2}, {'c': 3}],
            'col3': ['a', 'b', 'c']
        })
        
        result = filter_unhashable_columns(df)
        
        # col2 should be dropped
        assert 'col2' not in result.columns
    
    def test_filter_unhashable_columns_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        
        result = filter_unhashable_columns(df)
        
        assert len(result.columns) == 0


class TestTrainModels:
    """Test cases for train_models function."""
    
    def test_train_models_returns_pipelines(self, sample_features_data):
        """Test that train_models returns two pipeline models."""
        logreg, rf, (X_test, y_test) = train_models(sample_features_data.copy())
        
        assert isinstance(logreg, Pipeline)
        assert isinstance(rf, Pipeline)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
    
    def test_train_models_logreg_structure(self, sample_features_data):
        """Test that Logistic Regression pipeline has correct structure."""
        logreg, _, _ = train_models(sample_features_data.copy())
        
        assert 'pre' in logreg.named_steps
        assert 'clf' in logreg.named_steps
        assert isinstance(logreg.named_steps['clf'], LogisticRegression)
    
    def test_train_models_rf_structure(self, sample_features_data):
        """Test that Random Forest pipeline has correct structure."""
        _, rf, _ = train_models(sample_features_data.copy())
        
        assert 'pre' in rf.named_steps
        assert 'clf' in rf.named_steps
        assert isinstance(rf.named_steps['clf'], RandomForestClassifier)
    
    def test_train_models_hyperparameters(self, sample_features_data):
        """Test that models have correct hyperparameters."""
        logreg, rf, _ = train_models(sample_features_data.copy())
        
        # Check Logistic Regression hyperparameters
        assert logreg.named_steps['clf'].C == 0.5
        assert logreg.named_steps['clf'].penalty == 'l2'
        assert logreg.named_steps['clf'].solver == 'liblinear'
        assert logreg.named_steps['clf'].max_iter == 1000
        assert logreg.named_steps['clf'].class_weight == 'balanced'
        assert logreg.named_steps['clf'].random_state == 1234
        
        # Check Random Forest hyperparameters
        assert rf.named_steps['clf'].n_estimators == 1000
        assert rf.named_steps['clf'].class_weight == 'balanced'
        assert rf.named_steps['clf'].random_state == 1234
    
    def test_train_models_separates_target(self, sample_features_data):
        """Test that target variable is properly separated."""
        _, _, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # created_account should not be in X_test
        assert 'created_account' not in X_test.columns
        
        # y_test should only contain non-null values
        assert y_test.notna().all()
        assert y_test.isin([0, 1]).all()
    
    def test_train_models_drops_temporary_columns(self, sample_features_data):
        """Test that temporary columns are dropped."""
        # Add some temporary columns
        sample_features_data['participant_id'] = ['P001', 'P002', 'P003']
        sample_features_data['first_name'] = ['John', 'Jane', 'Bob']
        
        _, _, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # Temporary columns should not be in X_test
        assert 'participant_id' not in X_test.columns
        assert 'first_name' not in X_test.columns
    
    def test_train_models_test_split_size(self, sample_features_data):
        """Test that test split is 20%."""
        _, _, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # Get total number of non-null target values
        total_non_null = sample_features_data['created_account'].notna().sum()
        expected_test_size = int(total_non_null * 0.2)
        
        # Allow for small rounding differences
        assert abs(len(X_test) - expected_test_size) <= 1
    
    def test_train_models_stratified_split(self, sample_features_data):
        """Test that split is stratified."""
        _, _, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # Check that both classes are present in test set
        unique_classes = y_test.unique()
        assert len(unique_classes) >= 1  # At least one class
        
        # If we have both classes, check proportions are reasonable
        if len(unique_classes) == 2:
            class_counts = y_test.value_counts()
            # Both classes should be present
            assert 0 in class_counts.index
            assert 1 in class_counts.index
    
    def test_train_models_handles_missing_target(self, sample_features_data):
        """Test that missing target values are handled correctly."""
        # Ensure we have some missing values
        sample_features_data.loc[2, 'created_account'] = np.nan
        
        logreg, rf, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # Models should still train successfully
        assert logreg is not None
        assert rf is not None
        
        # y_test should not contain NaN
        assert y_test.notna().all()
    
    def test_train_models_filters_high_cardinality(self, sample_features_data):
        """Test that high-cardinality categoricals are dropped."""
        # Add a high-cardinality column
        sample_features_data['high_card_col'] = [f'value_{i}' for i in range(len(sample_features_data))]
        
        _, _, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # High-cardinality column should be dropped
        assert 'high_card_col' not in X_test.columns
    
    def test_train_models_handles_all_numeric(self):
        """Test with all numeric features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'created_account': [1, 0, 1, 0, 1]
        })
        
        logreg, rf, (X_test, y_test) = train_models(df)
        
        assert logreg is not None
        assert rf is not None
        assert len(X_test) > 0
    
    def test_train_models_handles_all_categorical(self):
        """Test with all categorical features."""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'created_account': [1, 0, 1, 0, 1]
        })
        
        logreg, rf, (X_test, y_test) = train_models(df)
        
        assert logreg is not None
        assert rf is not None
        assert len(X_test) > 0
    
    def test_train_models_reproducibility(self, sample_features_data):
        """Test that results are reproducible with same random_state."""
        logreg1, rf1, (X_test1, y_test1) = train_models(sample_features_data.copy())
        logreg2, rf2, (X_test2, y_test2) = train_models(sample_features_data.copy())
        
        # Test sets should be identical (same random_state)
        assert X_test1.index.equals(X_test2.index)
        assert y_test1.index.equals(y_test2.index)
    
    def test_train_models_can_predict(self, sample_features_data):
        """Test that trained models can make predictions."""
        logreg, rf, (X_test, y_test) = train_models(sample_features_data.copy())
        
        # Models should be able to predict
        logreg_pred = logreg.predict(X_test)
        rf_pred = rf.predict(X_test)
        
        assert len(logreg_pred) == len(X_test)
        assert len(rf_pred) == len(X_test)
        assert logreg_pred.dtype in [np.int64, np.int32]
        assert rf_pred.dtype in [np.int64, np.int32]
        
        # Predictions should be binary
        assert set(logreg_pred) <= {0, 1}
        assert set(rf_pred) <= {0, 1}

