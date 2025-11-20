"""
Unit tests for feature_engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features, parse_salary


class TestParseSalary:
    """Test cases for parse_salary function."""
    
    def test_parse_salary_nan(self):
        """Test parsing NaN salary."""
        result = parse_salary(np.nan)
        
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] is None
    
    def test_parse_salary_gbp_range(self):
        """Test parsing GBP salary range."""
        result = parse_salary("£30000-40000")
        
        assert result.iloc[0] == 35000.0  # Average of range
        assert result.iloc[1] == "GBP"
    
    def test_parse_salary_single_value(self):
        """Test parsing single salary value."""
        result = parse_salary("£50000")
        
        assert result.iloc[0] == 50000.0
        assert result.iloc[1] == "GBP"
    
    def test_parse_salary_mur(self):
        """Test parsing MUR salary."""
        result = parse_salary("100000-200000 MUR")
        
        assert result.iloc[0] == 150000.0
        assert result.iloc[1] == "MUR"
    
    def test_parse_salary_monthly(self):
        """Test parsing monthly salary."""
        result = parse_salary("£3000 per month")
        
        assert result.iloc[0] == 36000.0  # 3000 * 12
        assert result.iloc[1] == "GBP"
    
    def test_parse_salary_weekly(self):
        """Test parsing weekly salary."""
        result = parse_salary("£500 per week")
        
        assert result.iloc[0] == 26000.0  # 500 * 52
        assert result.iloc[1] == "GBP"
    
    def test_parse_salary_no_numbers(self):
        """Test parsing salary with no numbers."""
        result = parse_salary("unknown")
        
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == "GBP"


class TestEngineerFeatures:
    """Test cases for engineer_features function."""
    
    def test_engineer_features_returns_dataframe(self, sample_merged_data):
        """Test that engineer_features returns a DataFrame."""
        result = engineer_features(sample_merged_data.copy())
        
        assert isinstance(result, pd.DataFrame)
    
    def test_target_variable_cleaning(self, sample_merged_data):
        """Test that created_account is properly cleaned."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that 'yes' becomes 1, 'no' becomes 0, None stays as NaN
        assert result['created_account'].iloc[0] == 1.0
        assert result['created_account'].iloc[1] == 0.0
        assert pd.isna(result['created_account'].iloc[2])
    
    def test_age_binning(self, sample_merged_data):
        """Test that age is binned correctly."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that age_group columns are created
        age_cols = [col for col in result.columns if col.startswith('age_') and '-' in col]
        assert len(age_cols) > 0
        
        # Check that one age group is 1 for each row
        for idx in range(len(result)):
            age_group_values = [result[col].iloc[idx] for col in age_cols]
            assert sum(age_group_values) == 1, f"Row {idx} should have exactly one age group"
    
    def test_education_one_hot_encoding(self, sample_merged_data):
        """Test that education is one-hot encoded."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that education columns are created
        edu_cols = [col for col in result.columns if col.startswith('edu_') and not col.startswith('edu_num_')]
        assert len(edu_cols) > 0
        
        # Original education column should be removed
        assert 'education' not in result.columns
    
    def test_town_frequency_encoding(self, sample_merged_data):
        """Test that town is frequency encoded."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that town_freq column exists
        assert 'town_freq' in result.columns
        
        # Original town column should be removed
        assert 'town' not in result.columns
        
        # Frequency should be between 0 and 1
        assert (result['town_freq'] >= 0).all()
        assert (result['town_freq'] <= 1).all()
    
    def test_sex_encoding(self, sample_merged_data):
        """Test that sex is encoded correctly."""
        result = engineer_features(sample_merged_data.copy())
        
        # Male should be 1, Female should be 0
        assert result['sex'].iloc[0] == 1  # Male
        assert result['sex'].iloc[1] == 0  # Female
    
    def test_categorical_one_hot_encoding(self, sample_merged_data):
        """Test that categorical variables are one-hot encoded."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that one-hot encoded columns exist
        assert any(col.startswith('religion_') for col in result.columns)
        assert any(col.startswith('relationship_') for col in result.columns)
        assert any(col.startswith('workclass_') for col in result.columns)
        assert any(col.startswith('race_') for col in result.columns)
        assert any(col.startswith('native_country_') for col in result.columns)
        
        # Original columns should be removed
        assert 'religion' not in result.columns
        assert 'relationship' not in result.columns
        assert 'workclass' not in result.columns
        assert 'race' not in result.columns
        assert 'native_country' not in result.columns
    
    def test_job_title_label_encoding(self, sample_merged_data):
        """Test that job_title is label encoded."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that job_title_encoded exists
        assert 'job_title_encoded' in result.columns
        
        # Original job_title should be removed
        assert 'job_title' not in result.columns
        
        # Encoded values should be integers
        assert pd.api.types.is_integer_dtype(result['job_title_encoded'])
    
    def test_demographic_quantile_binning(self, sample_merged_data):
        """Test that demographic_characteristic is quantile binned."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that demo_group_qcut exists
        assert 'demo_group_qcut' in result.columns
        
        # Original column should be removed
        assert 'demographic_characteristic' not in result.columns
        
        # Values should be categorical (G1, G2, etc.)
        assert result['demo_group_qcut'].dtype.name == 'category'
    
    def test_education_num_one_hot_encoding(self, sample_merged_data):
        """Test that education_num is one-hot encoded."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that edu_num columns exist
        edu_num_cols = [col for col in result.columns if col.startswith('edu_num_')]
        assert len(edu_num_cols) > 0
    
    def test_net_profit_calculation(self, sample_merged_data):
        """Test that net_profit is calculated correctly."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that net_profit exists
        assert 'net_profit' in result.columns
        assert 'net_profit_deskewed' in result.columns
        
        # Original columns should be removed
        assert 'capital_gain' not in result.columns
        assert 'capital_loss' not in result.columns
        
        # Net profit should be gain - loss
        # For row 1: 1000 - 0 = 1000
        # For row 2: 500 - 100 = 400
        assert result['net_profit'].iloc[1] == 1000
        assert result['net_profit'].iloc[2] == 400
    
    def test_salary_parsing(self, sample_merged_data):
        """Test that salary is parsed and converted."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that salary_value_gbp exists
        assert 'salary_value_gbp' in result.columns
        
        # Original salary_band should be removed
        assert 'salary_band' not in result.columns
        
        # Salary should be numeric
        assert pd.api.types.is_numeric_dtype(result['salary_value_gbp'])
    
    def test_fb_scaling(self, sample_merged_data):
        """Test that Facebook metrics are scaled."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that scaled columns exist
        assert 'familiarity_FB_scaled' in result.columns
        assert 'view_FB_scaled' in result.columns
        
        # Original columns should be removed
        assert 'familiarity_FB' not in result.columns
        assert 'view_FB' not in result.columns
    
    def test_employment_duration(self, sample_merged_data):
        """Test that employment duration is calculated."""
        result = engineer_features(sample_merged_data.copy())
        
        # Check that employment duration columns exist
        assert 'employment_duration_years' in result.columns
        assert 'employment_duration_years_deskewed' in result.columns
        
        # Original columns should be removed
        assert 'years_with_employer' not in result.columns
        assert 'months_with_employer' not in result.columns
        
        # Duration should be years + months/12
        # Row 0: 5 + 6/12 = 5.5
        assert abs(result['employment_duration_years'].iloc[0] - 5.5) < 0.01
    
    def test_engineer_features_preserves_index(self, sample_merged_data):
        """Test that index is preserved."""
        original_index = sample_merged_data.index
        result = engineer_features(sample_merged_data.copy())
        
        assert result.index.equals(original_index)
    
    def test_engineer_features_no_side_effects(self, sample_merged_data):
        """Test that original dataframe is not modified."""
        original = sample_merged_data.copy()
        result = engineer_features(sample_merged_data.copy())
        
        # Original should not have new columns
        assert 'town_freq' not in original.columns
        assert 'salary_value_gbp' not in original.columns

