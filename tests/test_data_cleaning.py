"""
Unit tests for data_cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import basic_cleaning, detect_numeric_strings


class TestBasicCleaning:
    """Test cases for basic_cleaning function."""
    
    def test_basic_cleaning_returns_dataframes(self, sample_campaign_data, sample_mortgage_data):
        """Test that basic_cleaning returns two DataFrames."""
        campaign, mortgage = basic_cleaning(sample_campaign_data.copy(), sample_mortgage_data.copy())
        
        assert isinstance(campaign, pd.DataFrame)
        assert isinstance(mortgage, pd.DataFrame)
    
    def test_campaign_full_name_clean(self, sample_campaign_data):
        """Test that full_name_clean is created correctly for campaign."""
        campaign, _ = basic_cleaning(sample_campaign_data.copy(), pd.DataFrame())
        
        assert 'full_name_clean' in campaign.columns
        assert campaign['full_name_clean'].iloc[0] == 'john smith'
        assert campaign['full_name_clean'].iloc[1] == 'jane doe'
    
    def test_campaign_age_numeric(self, sample_campaign_data):
        """Test that age is converted to numeric."""
        campaign, _ = basic_cleaning(sample_campaign_data.copy(), pd.DataFrame())
        
        assert pd.api.types.is_numeric_dtype(campaign['age'])
        assert campaign['age'].iloc[0] == 30
    
    def test_mortgage_name_cleaning(self, sample_mortgage_data):
        """Test that mortgage names are cleaned correctly."""
        _, mortgage = basic_cleaning(pd.DataFrame(), sample_mortgage_data.copy())
        
        assert 'full_name_clean' in mortgage.columns
        assert 'name_clean_temp' in mortgage.columns
        assert 'first_last' in mortgage.columns
        
        # Check that titles are removed
        assert 'mr' not in mortgage['full_name_clean'].iloc[0].lower()
        assert 'mrs' not in mortgage['full_name_clean'].iloc[1].lower()
    
    def test_mortgage_age_from_dob(self, sample_mortgage_data):
        """Test that age_from_dob is calculated correctly."""
        _, mortgage = basic_cleaning(pd.DataFrame(), sample_mortgage_data.copy())
        
        assert 'age_from_dob' in mortgage.columns
        assert 'dob_parsed' in mortgage.columns
        assert pd.api.types.is_numeric_dtype(mortgage['age_from_dob'])
        
        # Age should be approximately 2018 - birth_year
        # For 1988, should be around 30
        assert 29 <= mortgage['age_from_dob'].iloc[0] <= 31
    
    def test_mortgage_title_removal(self, sample_mortgage_data):
        """Test that titles are properly removed from names."""
        _, mortgage = basic_cleaning(pd.DataFrame(), sample_mortgage_data.copy())
        
        # Check that full_name_clean doesn't contain titles
        for name in mortgage['full_name_clean']:
            assert not any(title in name.lower() for title in ['mr', 'mrs', 'ms', 'miss', 'dr'])
    
    def test_basic_cleaning_preserves_original_data(self, sample_campaign_data, sample_mortgage_data):
        """Test that original data is preserved (not modified in place)."""
        campaign_orig = sample_campaign_data.copy()
        mortgage_orig = sample_mortgage_data.copy()
        
        campaign, mortgage = basic_cleaning(campaign_orig.copy(), mortgage_orig.copy())
        
        # Original dataframes should not have new columns
        assert 'full_name_clean' not in sample_campaign_data.columns
        assert 'full_name_clean' not in sample_mortgage_data.columns


class TestDetectNumericStrings:
    """Test cases for detect_numeric_strings function."""
    
    def test_detect_numeric_strings_all_numeric(self):
        """Test with series containing all numeric strings."""
        series = pd.Series(['1', '2', '3', '4', '5'])
        result = detect_numeric_strings(series)
        
        assert result == 1.0
    
    def test_detect_numeric_strings_mixed(self):
        """Test with series containing mixed numeric and non-numeric."""
        series = pd.Series(['1', '2', 'abc', '4', '5'])
        result = detect_numeric_strings(series)
        
        assert result == 0.8  # 4 out of 5 are numeric
    
    def test_detect_numeric_strings_all_non_numeric(self):
        """Test with series containing all non-numeric strings."""
        series = pd.Series(['abc', 'def', 'ghi'])
        result = detect_numeric_strings(series)
        
        assert result == 0.0
    
    def test_detect_numeric_strings_with_nan(self):
        """Test with series containing NaN values."""
        series = pd.Series(['1', '2', np.nan, '4', '5'])
        result = detect_numeric_strings(series)
        
        # NaN values should be ignored
        assert result == 1.0
    
    def test_detect_numeric_strings_negative_numbers(self):
        """Test with series containing negative numbers."""
        series = pd.Series(['-1', '-2', '3', '4'])
        result = detect_numeric_strings(series)
        
        assert result == 1.0
    
    def test_detect_numeric_strings_decimal_numbers(self):
        """Test with series containing decimal numbers."""
        series = pd.Series(['1.5', '2.7', '3.0', '4'])
        result = detect_numeric_strings(series)
        
        assert result == 1.0
    
    def test_detect_numeric_strings_empty_series(self):
        """Test with empty series."""
        series = pd.Series([])
        result = detect_numeric_strings(series)
        
        assert result == 0.0

