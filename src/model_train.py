"""
Model training module for insurance account creation prediction.

This module implements the model training pipeline as per notebook 02_EDA_Campaign_Mortgage.ipynb.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def filter_unhashable_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out columns that contain unhashable types (like lists) 
    that can't be used with SimpleImputer strategy='most_frequent'.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with unhashable columns removed
    """
    cols_to_drop = []
    for col in X.columns:
        # Check if column contains lists or other unhashable types
        sample = X[col].dropna()
        if len(sample) > 0:
            # Check first non-null value
            first_val = sample.iloc[0]
            if isinstance(first_val, (list, dict, set)):
                cols_to_drop.append(col)
    
    if cols_to_drop:
        print(f"Dropping columns with unhashable types: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)
    
    return X


def _compute_test_size(n_rows: int, fraction: float = 0.2) -> int:
    if n_rows < 2:
        raise ValueError("Need at least 2 rows with non-null targets to train models.")
    size = max(1, int(round(n_rows * fraction)))
    if size >= n_rows:
        size = n_rows - 1
    return size


def _can_stratify(y: pd.Series, test_size: int) -> bool:
    if y.nunique() < 2:
        return False
    counts = y.value_counts()
    if (counts < 2).any():
        return False
    n_classes = len(counts)
    if test_size < n_classes or (len(y) - test_size) < n_classes:
        return False
    return True


def train_models(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, Tuple[pd.DataFrame, pd.Series]]:
    df = df.copy()
    
    # Separate target_df (missing created_account) and clean_df (non-missing)
    # (as per notebook Cell 47)
    target_df = df[df['created_account'].isnull()].copy()
    clean_df = df[df['created_account'].notnull()].copy()
    
    print(f"Total rows: {len(df)}, Training rows (non-null target): {len(clean_df)}, Prediction rows (null target): {len(target_df)}")
    
    # Use only clean_df for training (as per notebook)
    y = pd.to_numeric(clean_df['created_account'], errors='coerce').astype(int)
    X = clean_df.drop(columns=['created_account'])
    
    # Drop any remaining temporary/helper columns that shouldn't be used for modeling
    # (Note: Most should already be dropped in run_pipeline.py, but check for any remaining)
    temp_cols = ['participant_id', 'name_title', 'first_name', 'last_name', 'postcode',
                 'company_email', 'full_name_clean', 'full_name', 'dob', 'paye',
                 'name_clean_temp', 'first_last', 'dob_parsed', 'age_from_dob', 'new_mortgage']
    cols_to_drop = [col for col in temp_cols if col in X.columns]
    if cols_to_drop:
        print(f"Dropping remaining temporary columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)
    
    # Filter out columns with unhashable types (lists, dicts, etc.)
    X = filter_unhashable_columns(X)

    # Identify column types (as per notebook Cell 54)
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    
    # Drop high-cardinality categoricals BEFORE train_test_split (as per notebook Cell 54)
    # Columns with more than 50 unique values will be dropped
    MAX_CATEGORIES = 50
    high_card_cols = []
    for col in cat_cols:
        unique_count = X[col].nunique(dropna=True)
        unique_ratio = unique_count / max(len(X), 1)
        if unique_count > MAX_CATEGORIES or unique_ratio > 0.5:
            high_card_cols.append(col)
    if high_card_cols:
        print(f"Dropping high-cardinality categorical columns (> {MAX_CATEGORIES} unique values): {high_card_cols}")
        X = X.drop(columns=high_card_cols)
        cat_cols = [col for col in cat_cols if col not in high_card_cols]
    
    # Re-identify column types after dropping high-cardinality columns
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    # Define preprocessing (as per notebook Cell 54)
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if num_cols:
        transformers.append(('num', numeric_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers) if transformers else 'passthrough'

    # Split AFTER dropping high-cardinality columns (as per notebook Cell 59)
    total_rows = len(clean_df)
    test_size = _compute_test_size(total_rows)

    if total_rows <= 3:
        # Tiny dataset fallback: train on all labelled rows and reserve a small hold-out slice
        X_test = X.head(test_size).copy()
        y_test = y.loc[X_test.index].copy()
        X_train = X.copy()
        y_train = y.copy()
    else:
        stratify = y if _can_stratify(y, test_size) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify, random_state=1234
        )

        # If stratification couldn't preserve both classes in training, move one sample from
        # the test set to the training set (and replace it with another sample) to avoid
        # single-class training failures.
        if y_train.nunique() < 2 and y_test.nunique() >= 1:
            missing_classes = set(y_test.unique()) - set(y_train.unique())
            if missing_classes:
                class_to_add = next(iter(missing_classes))
                idx_to_move = y_test[y_test == class_to_add].index[0]
                X_train = pd.concat([X_train, X_test.loc[[idx_to_move]]])
                y_train = pd.concat([y_train, y_test.loc[[idx_to_move]]])
                X_test = X_test.drop(index=idx_to_move)
                y_test = y_test.drop(index=idx_to_move)

                # Keep test size by moving the earliest training sample to test if needed
                if len(X_test) < test_size and len(X_train) > 1:
                    replacement_idx = X_train.index[0]
                    X_test = pd.concat([X_test, X_train.loc[[replacement_idx]]])
                    y_test = pd.concat([y_test, y_train.loc[[replacement_idx]]])
                    X_train = X_train.drop(index=replacement_idx)
                    y_train = y_train.drop(index=replacement_idx)


    # Logistic Regression (as per notebook Cell 60)
    logreg = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(
        C=0.5,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=1234
    ))
    ])
    logreg.fit(X_train, y_train)


    # Random Forest (as per notebook Cell 61)
    rf = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=1000,
        class_weight='balanced',
        random_state=1234,
        n_jobs=-1
    ))
    ])
    rf.fit(X_train, y_train)


    return logreg, rf, (X_test, y_test)