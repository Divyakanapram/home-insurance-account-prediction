"""
Data loading module.

Loads campaign and mortgage datasets from CSV files.
"""

from typing import Tuple
import pandas as pd
from src.config import CAMPAIGN_FILE, MORTGAGE_FILE


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load campaign and mortgage datasets.
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (campaign, mortgage) dataframes
    """
    campaign = pd.read_csv(CAMPAIGN_FILE, dtype=str)
    mortgage = pd.read_csv(MORTGAGE_FILE, dtype=str)
    return campaign, mortgage