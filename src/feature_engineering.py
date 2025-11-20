"""
Feature engineering module.

Implements all feature transformations as per notebook 02_EDA_Campaign_Mortgage.ipynb.
"""

from typing import Optional
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def parse_salary(s) -> pd.Series:
    """
    Parse salary string into numeric value and currency.
    Matches notebook Cell 37 logic.
    
    Parameters:
    -----------
    s : str or NaN
        Salary string (e.g., "£50000-60000", "1000-2000 MUR per month")
        
    Returns:
    --------
    pd.Series : [value, currency]
        value: float or NaN
        currency: str (GBP, MUR, etc.)
    """
    if pd.isna(s):
        return pd.Series([np.nan, None], index=['salary_value', 'salary_currency'], dtype=object)

    s = str(s).lower().strip()
    s_clean = s.replace("£", "").replace(",", "")

    currency = "GBP"
    if "mur" in s_clean:
        currency = "MUR"
        s_clean = s_clean.replace("mur", "")

    nums = re.findall(r"\d+\.?\d*", s_clean)
    if not nums:
        return pd.Series([np.nan, currency], index=['salary_value', 'salary_currency'], dtype=object)

    value = (float(nums[0]) + float(nums[1])) / 2 if len(nums) >= 2 else float(nums[0])

    if "month" in s:
        value *= 12
    elif "pw" in s or "week" in s:
        value *= 52

    return pd.Series([value, currency], index=['salary_value', 'salary_currency'], dtype=object)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Matches notebook Cells 15-50 logic:
    - Target variable cleaning
    - Age binning
    - One-hot encoding
    - Frequency encoding
    - Salary parsing
    - Employment duration calculation
    - Net profit calculation
    - Feature scaling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset after column dropping
        
    Returns:
    --------
    pd.DataFrame
        Dataset with engineered features
    """
    df = df.copy()

    # --------------------------
    # TARGET CLEANING (as per notebook Cells 15-16, 36, 45)
    # --------------------------
    # First normalize strings (as per notebook Cell 15-16)
    df['created_account'] = df.get('created_account', np.nan)
    df['created_account'] = df['created_account'].astype(str).str.strip().str.lower()
    # Then map 'yes' to 1, 'no' to 0, keep NaN (which becomes 'nan' string) as NaN
    # Convert 'nan' string back to actual NaN
    df['created_account'] = df['created_account'].replace('nan', np.nan)
    df['created_account'] = df['created_account'].map({'yes': 1, 'no': 0})

    # --------------------------
    # AGE BINNING
    # --------------------------
    bins = list(range(10, 101, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    dummy_targets = {
        'education': 'edu',
        'age_group': 'age',
        'marital_status': 'marital'
    }
    for col, prefix in dummy_targets.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

    # --------------------------
    # TOWN FREQ ENC
    # --------------------------
    if 'town' in df.columns:
        town_freq = df['town'].value_counts(normalize=True)
        df['town_freq'] = df['town'].map(town_freq)
        df.drop(columns=['town'], errors='ignore', inplace=True)

    # --------------------------
    # SEX ENCODING (as per notebook Cell 27)
    # --------------------------
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    # --------------------------
    # ONE HOT ENCODE MULTI COLS
    # --------------------------
    for col in ['religion', 'relationship', 'workclass', 'race', 'native_country']:
        if col in df.columns:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc = ohe.fit_transform(df[[col]]).astype(int)
            enc_df = pd.DataFrame(
                enc,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                index=df.index
            )
            df = pd.concat([df, enc_df], axis=1)
            df.drop(columns=[col], errors='ignore', inplace=True)

    # --------------------------
    # LABEL ENCODE JOB TITLE
    # --------------------------
    if 'job_title' in df.columns:
        le = LabelEncoder()
        df['job_title_encoded'] = le.fit_transform(df['job_title'].astype(str))
        df.drop(columns=['job_title'], errors='ignore', inplace=True)

    # --------------------------
    # DEMOGRAPHIC CHARACTERISTIC QUANTILE BINNING
    # --------------------------
    if 'demographic_characteristic' in df.columns:
        df['demographic_characteristic'] = pd.to_numeric(df['demographic_characteristic'], errors='coerce')
        non_null = df['demographic_characteristic'].dropna()
        if non_null.nunique() >= 2:
            buckets = min(8, non_null.nunique())
            df['demo_group_qcut'] = pd.qcut(
                df['demographic_characteristic'],
                q=buckets,
                duplicates='drop'
            )
        else:
            df['demo_group_qcut'] = pd.Series(
                pd.Categorical(['Unknown'] * len(df)),
                index=df.index
            )
        df.drop(columns=['demographic_characteristic'], errors='ignore', inplace=True)
    # --------------------------
    # EDUCATION NUM OHE
    # --------------------------
    if 'education_num' in df.columns:
        df = pd.get_dummies(df, columns=['education_num'], prefix='edu_num', dtype=int)

    # --------------------------
    # NET PROFIT
    # --------------------------
    if 'capital_gain' not in df.columns:
        df['capital_gain'] = 0
    if 'capital_loss' not in df.columns:
        df['capital_loss'] = 0

    df['capital_gain'] = pd.to_numeric(df['capital_gain'], errors='coerce').fillna(0)
    df['capital_loss'] = pd.to_numeric(df['capital_loss'], errors='coerce').fillna(0)

    df['net_profit'] = df['capital_gain'] - df['capital_loss']
    df.drop(columns=['capital_gain', 'capital_loss'], errors='ignore', inplace=True)

    df['net_profit_deskewed'] = np.sqrt(df['net_profit'].clip(lower=0))

    # --------------------------
    # SALARY PARSING (as per notebook Cell 37)
    # --------------------------
    if 'salary_band' in df.columns:
        df[['salary_value', 'salary_currency']] = df['salary_band'].apply(parse_salary)
        df.drop(columns=['salary_band'], errors='ignore', inplace=True)

        # Define conversion rates (as per notebook Cell 37)
        conversion_rates = {'GBP': 1, 'USD': 0.81, 'EUR': 0.88, 'MUR': 0.018}
        
        # Convert salary_value to GBP (as per notebook Cell 37)
        df['salary_value_gbp'] = df.apply(
            lambda row: row['salary_value'] * conversion_rates.get(row['salary_currency'], 1),
            axis=1
        )
        df['salary_currency'] = 'GBP'
        df.drop(columns=['salary_currency', 'salary_value'], errors='ignore', inplace=True)

    # --------------------------
    # FB SCALING (SAFE DROP)
    # --------------------------
    if set(['familiarity_FB', 'view_FB']).issubset(df.columns):
        scaler = StandardScaler()
        df[['familiarity_FB_scaled', 'view_FB_scaled']] = scaler.fit_transform(
            df[['familiarity_FB', 'view_FB']]
        )
        df.drop(columns=['familiarity_FB', 'view_FB'], errors='ignore', inplace=True)

    # --------------------------
    # EMPLOYMENT DURATION
    # --------------------------
    if 'years_with_employer' in df.columns:
        df['years_with_employer'] = pd.to_numeric(df['years_with_employer'], errors='coerce')
    if 'months_with_employer' in df.columns:
        df['months_with_employer'] = pd.to_numeric(df['months_with_employer'], errors='coerce')

    if set(['years_with_employer', 'months_with_employer']).issubset(df.columns):
        df['employment_duration_years'] = (
            df['years_with_employer'] + df['months_with_employer'] / 12
        )
        df.drop(columns=['years_with_employer', 'months_with_employer'], errors='ignore', inplace=True)

        df['employment_duration_years'] = df['employment_duration_years'].clip(
            upper=df['employment_duration_years'].quantile(0.95)
        )
        df['employment_duration_years_deskewed'] = np.sqrt(
            df['employment_duration_years'].clip(lower=0)
        )

    return df
