"""
Data cleaning module.

Provides functions for cleaning and preparing campaign and mortgage datasets
for merging and feature engineering.
"""

from typing import Tuple
import pandas as pd
import numpy as np
import re


def basic_cleaning(
    campaign: pd.DataFrame,
    mortgage: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and prepare campaign and mortgage datasets for merging.
    
    Creates standardized name fields and calculates age from DOB.
    Matches notebook Cells 6-7 logic.
    
    Parameters:
    -----------
    campaign : pd.DataFrame
        Campaign dataset
    mortgage : pd.DataFrame
        Mortgage dataset
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (cleaned_campaign, cleaned_mortgage)
    """
    campaign = campaign.copy()
    mortgage = mortgage.copy()

    # Safely ensure columns exist before string ops
    for col in ['first_name', 'last_name']:
        if col not in campaign.columns:
            campaign[col] = ''
        campaign[col] = campaign[col].fillna('').astype(str)

    if 'age' not in campaign.columns:
        campaign['age'] = np.nan
    campaign['age'] = pd.to_numeric(campaign['age'], errors='coerce')

    campaign['full_name_clean'] = (
        campaign['first_name'].str.strip().str.lower() + ' ' +
        campaign['last_name'].str.strip().str.lower()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Mortgage cleaning
    title_regex = r"^(mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+"

    if 'full_name' not in mortgage.columns:
        mortgage['full_name'] = ''
    mortgage['full_name'] = mortgage['full_name'].fillna('').astype(str)

    mortgage['name_clean_temp'] = (
        mortgage['full_name']
        .str.lower().str.replace(r"\s+", " ", regex=True)
        .str.replace(title_regex, "", regex=True)
        .str.strip()
    )

    mortgage['first_last'] = mortgage['name_clean_temp'].str.split()
    mortgage['full_name_clean'] = mortgage['first_last'].apply(
        lambda tokens: np.nan if not tokens else f"{tokens[0]} {tokens[-1] if len(tokens) > 1 else tokens[0]}"
    )
    
    # Calculate age from DOB (as per notebook Cell 7)
    if 'dob' not in mortgage.columns:
        mortgage['dob'] = np.nan
    mortgage['dob_parsed'] = pd.to_datetime(mortgage['dob'], errors='coerce', dayfirst=True)
    mortgage['age_from_dob'] = 2018 - mortgage['dob_parsed'].dt.year
    mortgage['age_from_dob'] = pd.to_numeric(mortgage['age_from_dob'], errors='coerce')

    return campaign, mortgage

def detect_numeric_strings(series: pd.Series) -> float:
    """
    Check what percentage of non-null values look like numbers.
    
    Parameters:
    -----------
    series : pd.Series
        Series to check
        
    Returns:
    --------
    float : Percentage of numeric-like values (0.0 to 1.0)
    """
    if series is None:
        return 0.0
    non_null = series.dropna()
    if non_null.empty:
        return 0.0
    pattern = r'^-?\d+(\.\d+)?$'
    numeric_like = non_null.astype(str).str.match(pattern)
    if numeric_like.empty:
        return 0.0
    return float(numeric_like.mean())
