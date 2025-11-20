# Insurance Account Creation Prediction

A production-ready machine learning pipeline for predicting insurance account creation based on campaign and mortgage customer data. This repository contains the Responsible AI assignment for Lloyds Bank, focused on predicting customer interest in home insurance.

## Project Overview

This repository contains a complete end-to-end machine learning pipeline that:

1. **Ingests** campaign and mortgage datasets
2. **Cleans and merges** datasets using name and age matching
3. **Engineers features** through encoding, transformations, and feature creation
4. **Trains** two classification models (Logistic Regression and Random Forest)
5. **Evaluates** models with comprehensive metrics and explainability analysis

The pipeline is fully aligned with the analysis in `notebooks/02_EDA_Campaign_Mortgage.ipynb`.

## Data Sources

### Campaign Dataset (`data/campaign.csv`)
- **Size**: 32,060 rows × 16 columns
- **Contains**: Campaign interaction data, demographics, and target variable (`created_account`)
- **Key Features**: Age, education, marital status, job title, Facebook engagement metrics

### Mortgage Dataset (`data/mortgage.csv`)
- **Size**: 32,561 rows × 18 columns
- **Contains**: Customer financial and demographic information
- **Key Features**: Salary, employment duration, capital gains/losses, demographic characteristics

### Merged Dataset
- **Final Size**: 16,591 rows after inner join on name and age
- **Match Rate**: ~51.7% of campaign records successfully matched

## Key Findings from EDA

### Data Quality
- ✅ No duplicate rows in either dataset
- ✅ Most columns complete (except target variable)
- ⚠️ High missing rate in target variable (90.6% - expected for prediction task)
- ⚠️ 38.1% missing name titles (acceptable)

### Target Variable Distribution
- **Class 0 (No Account)**: 1,468 samples (91.4%)
- **Class 1 (Account Created)**: 136 samples (8.6%)
- **Severe class imbalance** - handled with `class_weight='balanced'`

### Key Insights
- **Age Groups**: 30-59 age groups show higher account creation rates (8-14%)
- **Education**: Higher education (Masters, Doctorate, Prof-school) correlates with account creation
- **Marital Status**: Married-civ-spouse has highest conversion rate (17.23%)
- **Geographic**: Edinburgh dominates with 59.6% of records

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd lloyds-insurance-prediction
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Optional Dependencies
For SHAP explainability (optional):
```bash
pip install shap
```

## Usage

### CLI Entry Points

Both CLI scripts expect the raw CSVs referenced in `src/config.py` (defaults: `data/campaign.csv` and `data/mortgage.csv`). Supply alternative paths by passing `--campaign-file` / `--mortgage-file` to the scripts or editing the config.

| Command | When to use | What it does | Key outputs |
| --- | --- | --- | --- |
| `python run_eda.py` | First look at the data or after refreshing the raw CSVs | Loads the campaign & mortgage files, prints the same diagnostics as the notebook, and mirrors every EDA plot programmatically | PNGs written to `output/eda_plots/` (missing values heatmap, unique counts, dataset size comparison, account-creation breakdowns) |
| `python run_pipeline.py` | Full model build and evaluation | Executes the entire production pipeline: load ➜ clean ➜ merge ➜ feature engineer ➜ train ➜ evaluate ➜ explain | Metrics in the console, feature-importance figures plus SHAP summaries in `output/explainability_plots/`, trained pipelines in-memory for immediate use |

### 1. Run Exploratory Data Analysis

```bash
python run_eda.py
```

This generates:
- Data quality reports
- Missing value analysis
- Visualization plots in `output/eda_plots/`
- Account-creation stacked bars (age/education/marital/etc.) in `output/eda_plots/`
- Logs noting any missing columns or empty merges

### 2. Run Full Pipeline

```bash
python run_pipeline.py
```

This executes:
1. Data loading and cleaning
2. Dataset merging
3. Feature engineering
4. Model training (Logistic Regression + Random Forest)
5. Model evaluation
6. Threshold optimization
7. Explainability analysis (feature importance + SHAP plots)

**Outputs**:
- Model evaluation metrics (printed to console)
- Feature importance plots: `output/explainability_plots/{model}_feature_importance.png`
- SHAP summary plots: `output/explainability_plots/{model}_shap_summary.png`
- Intermediate diagnostics (train/test split sizes, dropped helper columns, warnings if SHAP is unavailable)

### 3. Individual Module Usage

```python
from src.load_data import load_data
from src.data_cleaning import basic_cleaning
from src.feature_engineering import engineer_features
from src.model_train import train_models
from src.evaluation import evaluate_model

# Load and clean
campaign, mortgage = load_data()
campaign, mortgage = basic_cleaning(campaign, mortgage)

# Merge
merged = campaign.merge(
    mortgage,
    left_on=['full_name_clean', 'age'],
    right_on=['full_name_clean', 'age_from_dob'],
    how='inner',
    suffixes=('_camp', '_mort')
)

# Engineer features
df = engineer_features(merged)

# Train models
logreg, rf, (X_test, y_test) = train_models(df)

# Evaluate
evaluate_model(logreg, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")
```

## Model Performance

### Logistic Regression
- **ROC AUC**: 0.9699
- **Precision (Class 1)**: 0.50
- **Recall (Class 1)**: 0.85
- **F1-Score (Class 1)**: 0.63
- **Precision@50**: 0.48
- **Precision@100**: 0.27
- **Precision@200**: 0.135

### Random Forest
- **ROC AUC**: 0.9936
- **Precision (Class 1)**: 0.94
- **Recall (Class 1)**: 0.63
- **F1-Score (Class 1)**: 0.76
- **Precision@50**: 0.54
- **Precision@100**: 0.27
- **Precision@200**: 0.135

**Model Selection**: Random Forest provides better precision and overall performance, while Logistic Regression offers better recall for the positive class.

## Feature Engineering

### Transformations Applied

1. **Target Variable**: Normalized to binary (0/1), NaN preserved for prediction set
2. **Age**: Binned into 10-year groups (10-19, 20-29, ..., 90-99)
3. **Categorical Encoding**:
   - One-hot: education, age_group, marital_status, religion, relationship, workclass, race, native_country, education_num
   - Label: job_title
   - Frequency: town
4. **Numeric Features**:
   - Salary: Parsed and converted to annual GBP
   - Employment duration: Combined years + months, capped at 95th percentile, square root transformed
   - Net profit: capital_gain - capital_loss, capped at 97th percentile, square root transformed
   - Facebook metrics: StandardScaler applied
5. **Demographic**: Quantile-based binning into 8 groups

### Excluded Features
- Identifiers: participant_id, postcode, company_email, paye
- Name fields: first_name, last_name, full_name, etc.
- Date fields: dob, dob_parsed, age_from_dob
- High-cardinality categoricals (>50 unique values)

## Project Structure

```
lloyds-insurance-prediction/
├── data/
│   ├── campaign.csv          # Campaign dataset
│   └── mortgage.csv          # Mortgage dataset
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration paths
│   ├── load_data.py          # Data loading
│   ├── data_cleaning.py      # Data cleaning and merging prep
│   ├── feature_engineering.py # Feature transformations
│   ├── model_train.py        # Model training
│   ├── evaluation.py         # Model evaluation and explainability
│   └── eda.py                # Exploratory data analysis
├── notebooks/
│   └── 02_EDA_Campaign_Mortgage.ipynb  # Reference notebook
├── docs/
│   ├── data_quality_analysis.md
│   ├── model_card.md
│   └── requirements_coverage.md
├── output/
│   ├── eda_plots/            # EDA visualizations
│   └── explainability_plots/ # Model explainability plots
├── run_pipeline.py          # Main pipeline script
├── run_eda.py               # EDA script
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Model Configuration

### Hyperparameters

**Logistic Regression**:
- `C`: 0.5
- `penalty`: 'l2'
- `solver`: 'liblinear'
- `max_iter`: 1000
- `class_weight`: 'balanced'
- `random_state`: 1234

**Random Forest**:
- `n_estimators`: 1000
- `class_weight`: 'balanced'
- `random_state`: 1234
- `n_jobs`: -1

### Preprocessing
- **Numeric**: Median imputation + StandardScaler
- **Categorical**: Most frequent imputation + OneHotEncoder
- **Test Split**: 20% stratified, random_state=1234

## Evaluation Metrics

The pipeline evaluates models using:
- **Classification Report**: Precision, recall, F1-score per class
- **ROC AUC**: Overall model performance
- **Precision@K**: Top-K precision for ranking applications
- **Confusion Matrix**: Visual representation
- **Feature Importance**: Top features for interpretability
- **SHAP Values**: Model explainability (optional)

## Explainability

### Feature Importance
- **Logistic Regression**: Coefficient magnitudes
- **Random Forest**: Feature importances
- Plots saved to `output/explainability_plots/`

### SHAP Values
- **Random Forest**: TreeExplainer
- **Logistic Regression**: Explainer (linear)
- Summary plots visualize feature contributions

## Limitations

1. **Class Imbalance**: Severe imbalance (91.4% vs 8.6%) - handled with class weights
2. **Small Training Set**: Only ~1,604 labeled samples after filtering
3. **Missing Data**: 90.6% missing target values (expected for prediction task)
4. **Merge Rate**: Only 51.7% of campaign records matched with mortgage data
5. **Temporal Validity**: Model trained on historical data may not generalize to future periods

## Reproducibility

All random seeds are fixed:
- `random_state=1234` for train/test split and model training
- Ensures reproducible results across runs

## Documentation

- **Data Quality Analysis**: `docs/data_quality_analysis.md`
- **Model Card**: `docs/model_card.md`
- **Requirements Coverage**: `docs/requirements_coverage.md`
- **Reference Notebook**: `notebooks/02_EDA_Campaign_Mortgage.ipynb`

## Testing

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_engineering.py
```

### Test Coverage

The test suite includes:
- **Unit tests** for all modules (`tests/test_*.py`)
- **Integration tests** for the full pipeline (`tests/test_integration.py`)
- **Fixtures** for reusable test data (`tests/conftest.py`)

Test files:
- `test_load_data.py` - Data loading tests
- `test_data_cleaning.py` - Data cleaning tests
- `test_feature_engineering.py` - Feature engineering tests
- `test_model_train.py` - Model training tests
- `test_evaluation.py` - Evaluation tests
- `test_integration.py` - End-to-end pipeline tests

## Code Quality

- ✅ PEP-8 compliant
- ✅ Type hints throughout
- ✅ Modular, production-ready design
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ No unused code or dead paths
- ✅ Comprehensive unit test coverage

## Contact

**Maintainer**: Divya Thekke Kanapram
**Project**: Lloyds Bank Insurance Prediction

---

## Quick Start Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline
python run_pipeline.py

# 3. Check outputs
ls output/explainability_plots/
```
