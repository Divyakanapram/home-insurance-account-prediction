"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_campaign_data():
    """Sample campaign dataset for testing."""
    return pd.DataFrame({
        'participant_id': ['P001', 'P002', 'P003'],
        'name_title': ['Mr', 'Mrs', None],
        'first_name': ['John', 'Jane', 'Bob'],
        'last_name': ['Smith', 'Doe', 'Johnson'],
        'age': ['30', '25', '45'],
        'postcode': ['SW1A 1AA', 'M1 1AA', 'B1 1AA'],
        'marital_status': ['Married-civ-spouse', 'Never-married', 'Divorced'],
        'education': ['Bachelors', 'Masters', 'HS-grad'],
        'job_title': ['Engineer', 'Manager', 'Teacher'],
        'occupation_level': ['5', '7', '4'],
        'education_num': ['13', '14', '9'],
        'familiarity_FB': ['3', '5', '2'],
        'view_FB': ['10', '15', '5'],
        'interested_insurance': ['1', '1', '0'],
        'company_email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'created_account': ['yes', 'no', None]
    })


@pytest.fixture
def sample_mortgage_data():
    """Sample mortgage dataset for testing."""
    return pd.DataFrame({
        'full_name': ['Mr John Smith', 'Mrs Jane Doe', 'Bob Johnson'],
        'dob': ['01/01/1988', '15/06/1993', '20/03/1973'],
        'town': ['Edinburgh', 'Leeds', 'Birmingham'],
        'paye': ['12345', '67890', '11111'],
        'salary_band': ['£30000-40000', '£50000-60000', '£20000-30000'],
        'years_with_employer': ['5', '3', '10'],
        'months_with_employer': ['6', '0', '0'],
        'hours_per_week': ['40', '35', '40'],
        'capital_gain': ['0', '1000', '500'],
        'capital_loss': ['0', '0', '100'],
        'new_mortgage': ['No', 'Yes', 'No'],
        'sex': ['Male', 'Female', 'Male'],
        'religion': ['Christianity', 'Christianity', 'Not Stated'],
        'relationship': ['Husband', 'Wife', 'Not-in-family'],
        'race': ['White', 'White', 'Black'],
        'native_country': ['England', 'England', 'England'],
        'workclass': ['Private', 'Private', 'Local-gov'],
        'demographic_characteristic': ['100', '150', '80']
    })


@pytest.fixture
def sample_merged_data(sample_campaign_data, sample_mortgage_data):
    """Sample merged dataset for testing."""
    # Simulate merge result
    merged = pd.DataFrame({
        'age': [30, 25, 45],
        'marital_status': ['Married-civ-spouse', 'Never-married', 'Divorced'],
        'education': ['Bachelors', 'Masters', 'HS-grad'],
        'education_num': ['13', '14', '9'],
        'occupation_level': ['5', '7', '4'],
        'familiarity_FB': ['3', '5', '2'],
        'view_FB': ['10', '15', '5'],
        'interested_insurance': ['1', '1', '0'],
        'created_account': ['yes', 'no', None],
        'town': ['Edinburgh', 'Leeds', 'Birmingham'],
        'salary_band': ['£30000-40000', '£50000-60000', '£20000-30000'],
        'years_with_employer': ['5', '3', '10'],
        'months_with_employer': ['6', '0', '0'],
        'hours_per_week': ['40', '35', '40'],
        'capital_gain': ['0', '1000', '500'],
        'capital_loss': ['0', '0', '100'],
        'sex': ['Male', 'Female', 'Male'],
        'religion': ['Christianity', 'Christianity', 'Not Stated'],
        'relationship': ['Husband', 'Wife', 'Not-in-family'],
        'race': ['White', 'White', 'Black'],
        'native_country': ['England', 'England', 'England'],
        'workclass': ['Private', 'Private', 'Local-gov'],
        'demographic_characteristic': ['100', '150', '80'],
        'job_title': ['Engineer', 'Manager', 'Teacher']
    })
    return merged


@pytest.fixture
def sample_features_data():
    """Sample feature-engineered dataset for testing."""
    return pd.DataFrame({
        'age': [30, 25, 45],
        'occupation_level': [5, 7, 4],
        'familiarity_FB_scaled': [0.0, 1.0, -1.0],
        'view_FB_scaled': [0.0, 1.0, -1.0],
        'interested_insurance': [1, 1, 0],
        'created_account': [1.0, 0.0, np.nan],
        'sex': [1, 0, 1],
        'town_freq': [0.6, 0.05, 0.02],
        'salary_value_gbp': [35000, 55000, 25000],
        'net_profit': [0, 1000, 400],
        'net_profit_deskewed': [0, 31.6, 20.0],
        'employment_duration_years': [5.5, 3.0, 10.0],
        'employment_duration_years_deskewed': [2.35, 1.73, 3.16],
        'job_title_encoded': [0, 1, 2],
        'demo_group_qcut': ['G4', 'G6', 'G2'],
        'edu_Bachelors': [1, 0, 0],
        'edu_Masters': [0, 1, 0],
        'edu_HS-grad': [0, 0, 1],
        'age_30-39': [1, 0, 0],
        'age_20-29': [0, 1, 0],
        'age_40-49': [0, 0, 1],
        'marital_Married-civ-spouse': [1, 0, 0],
        'marital_Never-married': [0, 1, 0],
        'marital_Divorced': [0, 0, 1],
        'religion_Christianity': [1, 1, 0],
        'religion_Not Stated': [0, 0, 1],
        'relationship_Husband': [1, 0, 0],
        'relationship_Wife': [0, 1, 0],
        'relationship_Not-in-family': [0, 0, 1],
        'workclass_Private': [1, 1, 0],
        'workclass_Local-gov': [0, 0, 1],
        'race_White': [1, 1, 0],
        'race_Black': [0, 0, 1],
        'native_country_England': [1, 1, 1]
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create minimal test CSV files
    campaign_df = pd.DataFrame({
        'participant_id': ['P001'],
        'first_name': ['John'],
        'last_name': ['Smith'],
        'age': ['30']
    })
    mortgage_df = pd.DataFrame({
        'full_name': ['John Smith'],
        'dob': ['01/01/1988']
    })
    
    campaign_df.to_csv(data_dir / "campaign.csv", index=False)
    mortgage_df.to_csv(data_dir / "mortgage.csv", index=False)
    
    return data_dir

