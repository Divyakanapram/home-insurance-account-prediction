"""
Integration tests for the full pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.load_data import load_data
from src.data_cleaning import basic_cleaning
from src.feature_engineering import engineer_features
from src.model_train import train_models
from src.evaluation import evaluate_model


class TestFullPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_pipeline_end_to_end(self):
        """Test that the full pipeline runs without errors."""
        # Load data
        campaign, mortgage = load_data()
        
        assert len(campaign) > 0
        assert len(mortgage) > 0
        
        # Clean data
        campaign_clean, mortgage_clean = basic_cleaning(campaign.copy(), mortgage.copy())
        
        assert 'full_name_clean' in campaign_clean.columns
        assert 'full_name_clean' in mortgage_clean.columns
        assert 'age_from_dob' in mortgage_clean.columns
        
        # Merge
        merged = campaign_clean.merge(
            mortgage_clean,
            left_on=['full_name_clean', 'age'],
            right_on=['full_name_clean', 'age_from_dob'],
            how='inner',
            suffixes=('_camp', '_mort')
        )
        
        assert len(merged) > 0
        
        # Drop temporary columns
        columns_to_drop = [
            'participant_id', 'name_title', 'first_name', 'last_name', 'postcode',
            'company_email', 'full_name_clean', 'full_name', 'dob', 'paye',
            'name_clean_temp', 'first_last', 'dob_parsed', 'age_from_dob', 'new_mortgage'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in merged.columns]
        merged = merged.drop(columns=columns_to_drop)
        
        # Engineer features
        df = engineer_features(merged.copy())
        
        assert 'created_account' in df.columns
        assert len(df) > 0
        
        # Train models (only if we have enough labeled data)
        if df['created_account'].notna().sum() >= 10:
            logreg, rf, (X_test, y_test) = train_models(df.copy())
            
            assert logreg is not None
            assert rf is not None
            assert len(X_test) > 0
            assert len(y_test) > 0
            
            # Evaluate models
            logreg_pred, logreg_prob = evaluate_model(logreg, X_test, y_test, "Logistic Regression")
            rf_pred, rf_prob = evaluate_model(rf, X_test, y_test, "Random Forest")
            
            assert len(logreg_pred) == len(X_test)
            assert len(rf_pred) == len(X_test)
    
    def test_pipeline_data_types(self):
        """Test that data types are correct throughout pipeline."""
        # Load and clean
        campaign, mortgage = load_data()
        campaign_clean, mortgage_clean = basic_cleaning(campaign.copy(), mortgage.copy())
        
        # Check that age is numeric
        assert pd.api.types.is_numeric_dtype(campaign_clean['age'])
        assert pd.api.types.is_numeric_dtype(mortgage_clean['age_from_dob'])
        
        # Merge
        merged = campaign_clean.merge(
            mortgage_clean,
            left_on=['full_name_clean', 'age'],
            right_on=['full_name_clean', 'age_from_dob'],
            how='inner',
            suffixes=('_camp', '_mort')
        )
        
        # Engineer features
        df = engineer_features(merged.copy())
        
        # Check that created_account is numeric (0, 1, or NaN)
        if df['created_account'].notna().any():
            valid_values = df['created_account'].dropna()
            assert valid_values.isin([0.0, 1.0]).all()
    
    def test_pipeline_reproducibility(self):
        """Test that pipeline produces reproducible results."""
        # Load and process data
        campaign, mortgage = load_data()
        campaign_clean, mortgage_clean = basic_cleaning(campaign.copy(), mortgage.copy())
        
        merged = campaign_clean.merge(
            mortgage_clean,
            left_on=['full_name_clean', 'age'],
            right_on=['full_name_clean', 'age_from_dob'],
            how='inner',
            suffixes=('_camp', '_mort')
        )
        
        columns_to_drop = [
            'participant_id', 'name_title', 'first_name', 'last_name', 'postcode',
            'company_email', 'full_name_clean', 'full_name', 'dob', 'paye',
            'name_clean_temp', 'first_last', 'dob_parsed', 'age_from_dob', 'new_mortgage'
        ]
        columns_to_drop = [col for col in columns_to_drop if col in merged.columns]
        merged = merged.drop(columns=columns_to_drop)
        
        df1 = engineer_features(merged.copy())
        df2 = engineer_features(merged.copy())
        
        # Feature engineering should be deterministic
        # Check that key columns match
        if 'town_freq' in df1.columns and 'town_freq' in df2.columns:
            assert df1['town_freq'].equals(df2['town_freq'])
        
        # Train models with same data should produce same test split
        if df1['created_account'].notna().sum() >= 10:
            _, _, (X_test1, y_test1) = train_models(df1.copy())
            _, _, (X_test2, y_test2) = train_models(df2.copy())
            
            # Test sets should be identical (same random_state)
            assert X_test1.index.equals(X_test2.index)
            assert y_test1.index.equals(y_test2.index)

