"""
Exploratory Data Analysis (EDA) module for Campaign and Mortgage datasets.
Provides comprehensive analysis including data quality checks, statistics, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 160)
sns.set_style("whitegrid")

TITLE_REGEX = re.compile(r"^(mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+", re.IGNORECASE)
END_YEAR_FOR_AGE = 2018


def _normalize_mortgage_name(value):
    """Remove honorifics and keep first/last token."""
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = TITLE_REGEX.sub("", cleaned).strip()
    if not cleaned:
        return np.nan
    parts = cleaned.split(" ")
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else parts[0]
    return f"{first} {last}"


def _prepare_campaign_for_merge(df):
    df = df.copy()
    df['full_name_clean'] = (
        df['first_name'].fillna('').astype(str).str.strip().str.lower() + ' ' +
        df['last_name'].fillna('').astype(str).str.strip().str.lower()
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['full_name_clean', 'age'])
    return df


def _prepare_mortgage_for_merge(df):
    df = df.copy()
    df['full_name_clean'] = df['full_name'].apply(_normalize_mortgage_name)
    df['dob_parsed'] = pd.to_datetime(df['dob'], errors='coerce', dayfirst=True)
    dob_year = df['dob_parsed'].dt.year
    df['age_from_dob'] = END_YEAR_FOR_AGE - dob_year
    df['age_from_dob'] = pd.to_numeric(df['age_from_dob'], errors='coerce')
    df = df.dropna(subset=['full_name_clean', 'age_from_dob'])
    return df


def merge_datasets_for_account_analysis(campaign, mortgage):
    """
    Recreate the cleaned/merged dataset used inside the EDA notebook so that
    downstream visualizations (account creation breakdowns) can be automated.
    """
    campaign_prepped = _prepare_campaign_for_merge(campaign)
    mortgage_prepped = _prepare_mortgage_for_merge(mortgage)

    merged = pd.merge(
        campaign_prepped,
        mortgage_prepped,
        left_on=['full_name_clean', 'age'],
        right_on=['full_name_clean', 'age_from_dob'],
        how='inner',
        suffixes=('', '_mortgage')
    )

    drop_cols = [col for col in ['dob_parsed', 'age_from_dob'] if col in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    if merged.empty:
        print("⚠️  Unable to build merged campaign-mortgage dataset (no matching rows).")
    else:
        print(f"\nMerged dataset ready: {len(merged):,} rows after joining campaign & mortgage records.")

    return merged


def _map_created_account(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    mapping = {'yes': 1, 'y': 1, '1': 1, 'no': 0, 'n': 0, '0': 0}
    return mapping.get(value, np.nan)


def _age_sort_key(label):
    match = re.match(r'(\d+)-', str(label))
    if match:
        return int(match.group(1))
    return 999 if str(label).lower() == 'unknown' else 998


def _level_sort_key(label):
    match = re.search(r'(\d+)', str(label))
    if match:
        return int(match.group(1))
    return 999


def prepare_account_creation_features(merged_df):
    """
    Add the engineered columns that power the stacked bar charts from the notebook.
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    df = merged_df.copy()
    df['created_account_flag'] = df['created_account'].apply(_map_created_account)
    df = df[df['created_account_flag'].isin([0, 1])]
    if df.empty:
        return df

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    bins = list(range(10, 101, 10))
    labels = [f"{b}-{b+9}" for b in bins[:-1]]
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    df['age_group'] = df['age_group'].astype(str).replace('nan', 'Unknown')

    for col in ['education', 'marital_status', 'religion', 'relationship', 'workclass']:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown')

    df['occupation_level'] = pd.to_numeric(df['occupation_level'], errors='coerce')
    df['occupation_level_label'] = df['occupation_level'].apply(
        lambda x: f"Level {int(x)}" if pd.notna(x) else 'Unknown'
    )

    ins_map = {'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No'}
    if 'interested_insurance' in df.columns:
        df['interested_insurance_label'] = (
            df['interested_insurance']
            .astype(str)
            .str.strip()
            .str.lower()
            .map(ins_map)
            .fillna('Unknown')
        )
    else:
        df['interested_insurance_label'] = 'Unknown'

    if 'demographic_characteristic' in df.columns:
        df['demographic_characteristic'] = pd.to_numeric(
            df['demographic_characteristic'], errors='coerce'
        )
        demo_series = df['demographic_characteristic'].dropna()
        if demo_series.nunique() > 1:
            q = min(8, demo_series.nunique())
            df.loc[demo_series.index, 'demo_group_qcut'] = pd.qcut(
                demo_series, q=q, labels=[f"G{i+1}" for i in range(q)], duplicates='drop'
            ).astype(str)
    if 'demo_group_qcut' in df.columns:
        df['demo_group_qcut'] = df['demo_group_qcut'].fillna('Unknown')
    else:
        df['demo_group_qcut'] = 'Unknown'

    if 'town' in df.columns:
        df['town'] = df['town'].fillna('Unknown')
        town_freq = df['town'].value_counts(normalize=True)
        df['town_freq'] = df['town'].map(town_freq)
        freq_bins = [0, 0.01, 0.05, 0.1, 0.25, 1.0]
        freq_labels = ['<1%', '1-5%', '5-10%', '10-25%', '25%+']
        df['town_freq_bucket'] = pd.cut(
            df['town_freq'],
            bins=freq_bins,
            labels=freq_labels,
            include_lowest=True
        )
        df['town_freq_bucket'] = df['town_freq_bucket'].astype(str).replace('nan', 'Unknown')
    else:
        df['town_freq_bucket'] = 'Unknown'

    return df


def plot_percentage_by_category(df, column, ax, title, sort_key=None, top_n=None):
    """
    Reusable percentage stacked-bar plot (mirrors the notebook output).
    """
    required_cols = [column, 'created_account_flag']
    if any(col not in df.columns for col in required_cols):
        ax.axis('off')
        ax.set_title(f"No data for {title}")
        print(f"⚠️  {title}: source column '{column}' missing.")
        return

    subset = df[required_cols].dropna(subset=['created_account_flag']).copy()
    subset[column] = subset[column].fillna('Unknown').astype(str)
    subset = subset[subset[column] != '']

    if subset.empty:
        ax.axis('off')
        ax.set_title(f"No data for {title}")
        print(f"⚠️  {title}: no labelled rows to plot.")
        return

    counts = subset.groupby([column, 'created_account_flag']).size().unstack(fill_value=0)
    for value in [0, 1]:
        if value not in counts.columns:
            counts[value] = 0
    counts = counts[counts.sum(axis=1) > 0]
    if counts.empty:
        ax.axis('off')
        ax.set_title(f"No data for {title}")
        print(f"⚠️  {title}: insufficient counts after grouping.")
        return

    if sort_key:
        ordered_index = sorted(counts.index, key=sort_key)
    else:
        ordered_index = counts.sum(axis=1).sort_values(ascending=False).index.tolist()

    if top_n:
        ordered_index = ordered_index[:top_n]

    counts = counts.loc[ordered_index]
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.rename(columns={0: 'No %', 1: 'Yes %'})

    percentages[['No %', 'Yes %']].plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['salmon', 'seagreen']
    )
    ax.set_title(f'{title} (Percentage Breakdown)', fontweight='bold')
    ax.set_xlabel(column.replace('_', ' ').title())
    ax.set_ylabel('Percentage of Customers')
    ax.legend(title='Created Account', labels=['No (0)', 'Yes (1)'])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')

    summary = percentages.copy()
    summary['Yes Count'] = counts[1]
    print(f"\n📊 {title} — Percentage Breakdown with Counts")
    print(summary.round(2).to_string())
    print("-" * 60)



def detect_numeric_strings(series):
    """Check what % of non-null values look like numbers."""
    if series.isna().all():
        return 0.0
    pattern = r'^-?\d+(\.\d+)?$'
    numeric_like = series.dropna().astype(str).str.match(pattern)
    return numeric_like.mean()


def analyze_dataset(df, dataset_name="Dataset"):
    """
    Perform comprehensive EDA on a dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    dataset_name : str
        Name of the dataset for display
    
    Returns:
    --------
    dict : Dictionary containing analysis results
    """
    print("\n" + "="*70)
    print(f"EXPLORATORY DATA ANALYSIS: {dataset_name.upper()}")
    print("="*70)
    
    results = {}
    
    # 1. Basic Information
    print(f"\n{'='*70}")
    print("1. BASIC INFORMATION")
    print(f"{'='*70}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    results['shape'] = df.shape
    results['columns'] = df.columns.tolist()
    
    # 2. Data Types
    print(f"\n{'='*70}")
    print("2. DATA TYPES")
    print(f"{'='*70}")
    dtype_counts = df.dtypes.value_counts()
    print("\nData type distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    print("\nDetailed data types:")
    for col in df.columns:
        print(f"  {col:30s}: {str(df[col].dtype)}")
    
    results['dtypes'] = df.dtypes.to_dict()
    
    # 3. Missing Values
    print(f"\n{'='*70}")
    print("3. MISSING VALUES ANALYSIS")
    print(f"{'='*70}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print("\nMissing values summary:")
    print(missing_df[missing_df['Missing Count'] > 0])
    
    if missing.sum() == 0:
        print("\n✓ No missing values found!")
    else:
        print(f"\nTotal missing values: {missing.sum()}")
        print(f"Columns with missing values: {(missing > 0).sum()}")
    
    results['missing'] = missing_df
    
    # 4. Duplicate Rows
    print(f"\n{'='*70}")
    print("4. DUPLICATE ROWS")
    print(f"{'='*70}")
    n_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {n_duplicates}")
    if n_duplicates > 0:
        print(f"Percentage of duplicates: {(n_duplicates/len(df)*100):.2f}%")
    else:
        print("✓ No duplicate rows found!")
    
    results['duplicates'] = n_duplicates
    
    # 5. Numeric-like String Detection
    print(f"\n{'='*70}")
    print("5. NUMERIC-LIKE STRING DETECTION")
    print(f"{'='*70}")
    print("Columns that might be numeric but stored as strings (>80% numeric-like):")
    numeric_like_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            pct = detect_numeric_strings(df[col])
            if pct > 0.8:
                print(f"  {col:30s}: {pct*100:5.1f}% numeric-like → consider converting")
                numeric_like_cols.append((col, pct))
    
    if not numeric_like_cols:
        print("  No columns detected as numeric-like strings")
    
    results['numeric_like'] = numeric_like_cols
    
    # 6. Unique Values Analysis
    print(f"\n{'='*70}")
    print("6. UNIQUE VALUES ANALYSIS")
    print(f"{'='*70}")
    print("Unique value counts per column:")
    unique_counts = {}
    for col in df.columns:
        n_unique = df[col].nunique()
        unique_counts[col] = n_unique
        pct_unique = (n_unique / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {col:30s}: {n_unique:6d} unique ({pct_unique:5.1f}% of rows)")
    
    results['unique_counts'] = unique_counts
    
    # 7. Descriptive Statistics (for numeric-like columns)
    print(f"\n{'='*70}")
    print("7. DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")
    
    # Try to convert numeric-like columns
    df_numeric = df.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='ignore')
            except:
                pass
    
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("\nNumeric columns statistics:")
        print(df_numeric[numeric_cols].describe().T)
        results['numeric_stats'] = df_numeric[numeric_cols].describe()
    else:
        print("No numeric columns detected")
        results['numeric_stats'] = None
    
    # 8. Categorical Analysis
    print(f"\n{'='*70}")
    print("8. CATEGORICAL COLUMNS ANALYSIS")
    print(f"{'='*70}")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nTop values for categorical columns:")
        for col in categorical_cols[:10]:  # Limit to first 10 to avoid too much output
            print(f"\n  {col}:")
            value_counts = df[col].value_counts().head(10)
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"    {str(val)[:50]:50s}: {count:6d} ({pct:5.1f}%)")
        results['categorical_cols'] = list(categorical_cols)
    else:
        print("No categorical columns found")
        results['categorical_cols'] = []
    
    # 9. Sample Data Preview
    print(f"\n{'='*70}")
    print("9. SAMPLE DATA PREVIEW")
    print(f"{'='*70}")
    print("\nFirst 5 rows (transposed for better readability):")
    print(df.head().T)
    
    print("\nLast 5 rows (transposed):")
    print(df.tail().T)
    
    return results


def create_visualizations(campaign, mortgage, merged=None, output_dir="output/eda_plots"):
    """
    Create visualizations for both datasets.
    
    Parameters:
    -----------
    campaign : pd.DataFrame
        Campaign dataset
    mortgage : pd.DataFrame
        Mortgage dataset
    output_dir : str
        Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # 1. Missing Values Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Campaign missing values
    campaign_missing = campaign.isnull()
    if campaign_missing.sum().sum() > 0:
        sns.heatmap(campaign_missing, yticklabels=False, cbar=True, ax=axes[0], cmap='viridis')
        axes[0].set_title('Campaign Dataset - Missing Values')
    else:
        axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[0].set_title('Campaign Dataset - Missing Values')
    
    # Mortgage missing values
    mortgage_missing = mortgage.isnull()
    if mortgage_missing.sum().sum() > 0:
        sns.heatmap(mortgage_missing, yticklabels=False, cbar=True, ax=axes[1], cmap='viridis')
        axes[1].set_title('Mortgage Dataset - Missing Values')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[1].set_title('Mortgage Dataset - Missing Values')
    
    plt.tight_layout()
    heatmap_path = output_dir / "missing_values_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {heatmap_path}")
    plt.close()
    
    # 2. Unique Values Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    campaign_unique = pd.Series({col: campaign[col].nunique() for col in campaign.columns})
    mortgage_unique = pd.Series({col: mortgage[col].nunique() for col in mortgage.columns})
    
    campaign_unique.sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[0])
    axes[0].set_title('Campaign - Top 15 Columns by Unique Values')
    axes[0].set_xlabel('Number of Unique Values')
    
    mortgage_unique.sort_values(ascending=True).tail(15).plot(kind='barh', ax=axes[1])
    axes[1].set_title('Mortgage - Top 15 Columns by Unique Values')
    axes[1].set_xlabel('Number of Unique Values')
    
    plt.tight_layout()
    unique_path = output_dir / "unique_values_comparison.png"
    plt.savefig(unique_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {unique_path}")
    plt.close()
    
    # 3. Dataset Size Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rows comparison
    sizes = [len(campaign), len(mortgage)]
    axes[0].bar(['Campaign', 'Mortgage'], sizes, color=['skyblue', 'lightcoral'])
    axes[0].set_title('Number of Rows')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(sizes):
        axes[0].text(i, v, str(v), ha='center', va='bottom')
    
    # Columns comparison
    sizes = [len(campaign.columns), len(mortgage.columns)]
    axes[1].bar(['Campaign', 'Mortgage'], sizes, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Number of Columns')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(sizes):
        axes[1].text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    size_path = output_dir / "dataset_size_comparison.png"
    plt.savefig(size_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {size_path}")
    plt.close()
    
    if merged is not None and not merged.empty:
        create_account_creation_visualizations(merged, output_dir)
    else:
        print("\n⚠️  Skipping account-creation plots (merged dataset missing or empty).")

    print(f"\n✓ All visualizations saved to: {output_dir}/")


def create_account_creation_visualizations(merged_df, output_dir):
    """
    Port of the notebook's stacked bar charts that break down account creation
    by key demographic and behavioural attributes.
    """
    output_dir = Path(output_dir)
    account_df = prepare_account_creation_features(merged_df)
    if account_df.empty:
        print("\n⚠️  Skipping account-creation plots (insufficient labelled rows).")
        return

    print("\nGenerating account creation breakdown visualizations...")

    # 1. Age group & Education
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plot_percentage_by_category(
        account_df,
        'age_group',
        axes[0],
        'Account Creation by Age Group',
        sort_key=_age_sort_key
    )
    plot_percentage_by_category(
        account_df,
        'education',
        axes[1],
        'Account Creation by Education Level',
        top_n=15
    )
    plt.tight_layout()
    age_edu_path = output_dir / "account_creation_age_education.png"
    plt.savefig(age_edu_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {age_edu_path}")
    plt.close(fig)

    # 2. Marital status & Relationship
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8))
    plot_percentage_by_category(
        account_df,
        'marital_status',
        axes1[0],
        'Account Creation by Marital Status',
        top_n=10
    )
    plot_percentage_by_category(
        account_df,
        'relationship',
        axes1[1],
        'Account Creation by Relationship',
        top_n=10
    )
    plt.tight_layout()
    marital_path = output_dir / "account_creation_marital_relationship.png"
    plt.savefig(marital_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {marital_path}")
    plt.close(fig1)

    # 3. Religion & Workclass
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
    plot_percentage_by_category(
        account_df,
        'religion',
        axes2[0],
        'Account Creation by Religion'
    )
    plot_percentage_by_category(
        account_df,
        'workclass',
        axes2[1],
        'Account Creation by Workclass'
    )
    plt.tight_layout()
    religion_path = output_dir / "account_creation_religion_workclass.png"
    plt.savefig(religion_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {religion_path}")
    plt.close(fig2)

    # 4. Occupation, Insurance Interest, Demo Group, Town Frequency
    fig3, axes3 = plt.subplots(2, 2, figsize=(20, 18))
    axes3 = axes3.flatten()
    plot_percentage_by_category(
        account_df,
        'occupation_level_label',
        axes3[0],
        'Account Creation by Occupation Level',
        sort_key=_level_sort_key
    )
    plot_percentage_by_category(
        account_df,
        'interested_insurance_label',
        axes3[1],
        'Account Creation by Insurance Interest'
    )
    plot_percentage_by_category(
        account_df,
        'demo_group_qcut',
        axes3[2],
        'Account Creation by Demo Group'
    )
    plot_percentage_by_category(
        account_df,
        'town_freq_bucket',
        axes3[3],
        'Account Creation by Town Frequency Band'
    )
    plt.tight_layout()
    drivers_path = output_dir / "account_creation_other_drivers.png"
    plt.savefig(drivers_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {drivers_path}")
    plt.close(fig3)

def compare_datasets(campaign, mortgage):
    """
    Compare both datasets side by side.
    
    Parameters:
    -----------
    campaign : pd.DataFrame
        Campaign dataset
    mortgage : pd.DataFrame
        Mortgage dataset
    """
    print(f"\n{'='*70}")
    print("DATASET COMPARISON")
    print(f"{'='*70}")
    
    print("\n1. Size Comparison:")
    print(f"  Campaign: {campaign.shape[0]:,} rows × {campaign.shape[1]} columns")
    print(f"  Mortgage: {mortgage.shape[0]:,} rows × {mortgage.shape[1]} columns")
    
    print("\n2. Common Columns:")
    common_cols = set(campaign.columns) & set(mortgage.columns)
    if common_cols:
        for col in sorted(common_cols):
            print(f"  - {col}")
    else:
        print("  No common columns found")
    
    print("\n3. Campaign-only Columns:")
    campaign_only = set(campaign.columns) - set(mortgage.columns)
    if campaign_only:
        for col in sorted(campaign_only):
            print(f"  - {col}")
    else:
        print("  None")
    
    print("\n4. Mortgage-only Columns:")
    mortgage_only = set(mortgage.columns) - set(campaign.columns)
    if mortgage_only:
        for col in sorted(mortgage_only):
            print(f"  - {col}")
    else:
        print("  None")
    
    print("\n5. Data Type Comparison:")
    print("  Campaign dtypes:")
    for dtype, count in campaign.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")
    print("  Mortgage dtypes:")
    for dtype, count in mortgage.dtypes.value_counts().items():
        print(f"    {dtype}: {count}")


def run_full_eda(campaign_file=None, mortgage_file=None, save_plots=True):
    """
    Run complete EDA on both datasets.
    
    Parameters:
    -----------
    campaign_file : str, optional
        Path to campaign.csv. If None, uses config.
    mortgage_file : str, optional
        Path to mortgage.csv. If None, uses config.
    save_plots : bool
        Whether to save visualization plots
    """
    # Load data
    if campaign_file is None or mortgage_file is None:
        from src.config import CAMPAIGN_FILE, MORTGAGE_FILE
        campaign_file = campaign_file or CAMPAIGN_FILE
        mortgage_file = mortgage_file or MORTGAGE_FILE
    
    print("Loading datasets...")
    campaign = pd.read_csv(campaign_file, dtype=str)
    mortgage = pd.read_csv(mortgage_file, dtype=str)
    
    # Analyze each dataset
    campaign_results = analyze_dataset(campaign, "Campaign")
    mortgage_results = analyze_dataset(mortgage, "Mortgage")

    # Build merged dataset for joint EDA (mirrors notebook workflow)
    merged_data = merge_datasets_for_account_analysis(campaign, mortgage)
    
    # Compare datasets
    compare_datasets(campaign, mortgage)
    
    # Create visualizations
    if save_plots:
        create_visualizations(campaign, mortgage, merged_data)
    
    print(f"\n{'='*70}")
    print("EDA COMPLETE!")
    print(f"{'='*70}")
    
    return {
        'campaign': campaign,
        'mortgage': mortgage,
        'merged': merged_data,
        'campaign_results': campaign_results,
        'mortgage_results': mortgage_results
    }


if __name__ == "__main__":
    # Run EDA when script is executed directly
    results = run_full_eda()
