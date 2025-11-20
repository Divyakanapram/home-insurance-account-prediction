"""
Unit tests for load_data module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, mock_open
import os
from src.load_data import load_data
from src.config import CAMPAIGN_FILE, MORTGAGE_FILE


class TestLoadData:
    """Test cases for load_data function."""
    
    def test_load_data_returns_dataframes(self):
        """Test that load_data returns two DataFrames."""
        campaign, mortgage = load_data()
        
        assert isinstance(campaign, pd.DataFrame)
        assert isinstance(mortgage, pd.DataFrame)
    
    def test_load_data_campaign_columns(self):
        """Test that campaign DataFrame has expected columns."""
        campaign, _ = load_data()
        
        expected_columns = [
            'participant_id', 'name_title', 'first_name', 'last_name', 'age',
            'postcode', 'marital_status', 'education', 'job_title',
            'occupation_level', 'education_num', 'familiarity_FB', 'view_FB',
            'interested_insurance', 'company_email', 'created_account'
        ]
        
        assert all(col in campaign.columns for col in expected_columns)
    
    def test_load_data_mortgage_columns(self):
        """Test that mortgage DataFrame has expected columns."""
        _, mortgage = load_data()
        
        expected_columns = [
            'full_name', 'dob', 'town', 'paye', 'salary_band',
            'years_with_employer', 'months_with_employer', 'hours_per_week',
            'capital_gain', 'capital_loss', 'new_mortgage', 'sex', 'religion',
            'relationship', 'race', 'native_country', 'workclass',
            'demographic_characteristic'
        ]
        
        assert all(col in mortgage.columns for col in expected_columns)
    
    def test_load_data_dtype_str(self):
        """Test that all columns are loaded as strings."""
        campaign, mortgage = load_data()
        
        # Check campaign columns
        for col in campaign.columns:
            assert campaign[col].dtype == 'object', f"Column {col} is not object type"
        
        # Check mortgage columns
        for col in mortgage.columns:
            assert mortgage[col].dtype == 'object', f"Column {col} is not object type"
    
    def test_load_data_non_empty(self):
        """Test that loaded DataFrames are not empty."""
        campaign, mortgage = load_data()
        
        assert len(campaign) > 0, "Campaign DataFrame is empty"
        assert len(mortgage) > 0, "Mortgage DataFrame is empty"
    
    @patch('pandas.read_csv')
    def test_load_data_file_paths(self, mock_read_csv):
        """Test that load_data uses correct file paths."""
        mock_read_csv.return_value = pd.DataFrame()
        
        load_data()
        
        # Verify read_csv was called with correct paths
        assert mock_read_csv.call_count == 2
        call_args = [call[0][0] for call in mock_read_csv.call_args_list]
        assert CAMPAIGN_FILE in call_args
        assert MORTGAGE_FILE in call_args
    
    @patch('pandas.read_csv')
    def test_load_data_dtype_parameter(self, mock_read_csv):
        """Test that load_data passes dtype=str to read_csv."""
        mock_read_csv.return_value = pd.DataFrame()
        
        load_data()
        
        # Check that dtype=str was passed
        for call in mock_read_csv.call_args_list:
            assert call[1]['dtype'] == str

