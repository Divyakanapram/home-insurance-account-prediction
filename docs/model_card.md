# Model Card: Insurance Account Creation Prediction

## Model Details

### Model Name
Insurance Account Creation Prediction Model

### Model Version
1.0.0

### Model Date
20-11-2025

### Model Type
Binary Classification

### Model Architecture
Two models are provided:
1. **Logistic Regression** - Linear model for interpretability
2. **Random Forest** - Ensemble model for higher performance

### Model Purpose
Predict the likelihood that a customer will create an insurance account after a marketing campaign. This model helps identify high-probability customers for targeted marketing efforts.

---

## Intended Use

### Primary Use Case
- **Target Audience**: Marketing department, campaign managers
- **Use**: Identify customers most likely to create an insurance account
- **Decision Support**: Prioritize marketing outreach and resource allocation

### Out-of-Scope Uses
- **NOT for**: Automated account creation decisions
- **NOT for**: Credit scoring or financial risk assessment
- **NOT for**: Real-time individual customer predictions without human review

### Ethical Considerations
- Model predictions should not be the sole basis for marketing decisions
- Consider fairness across demographic groups
- Regular monitoring required to detect model drift

---

## Training Data

### Data Sources
1. **Campaign Dataset** (`campaign.csv`)
   - 32,060 rows × 16 columns
   - Contains campaign interaction data, demographics, and target variable
   
2. **Mortgage Dataset** (`mortgage.csv`)
   - 32,561 rows × 18 columns
   - Contains customer financial and demographic information

### Data Collection
- **Collection Method**: --
- **Collection Date**: --
- **Data Period**: --

### Data Preprocessing
- Merged datasets on `full_name_clean` and `age` (inner join)
- Removed rows with missing target variable (created_account)
- Final training dataset: ~1,604 rows with non-null target
- Test set: 20% of training data (stratified split)

### Data Quality
- **Missing Values**: 
  - Campaign: `created_account` (90.6% missing), `name_title` (38.1% missing)
  - Mortgage: No missing values
- **Data Types**: All columns initially loaded as strings, converted to appropriate types
- **Duplicates**: No duplicate rows detected

### Training/Validation/Test Split
- **Training Set**: 80% (stratified)
- **Test Set**: 20% (stratified)
- **Random State**: 1234 (for reproducibility, as per notebook Cell 59)
- **Stratification**: Yes (to maintain class distribution)

---

## Features

### Feature Engineering
The model uses the following feature transformations:

1. **Demographic Features**:
   - Age (binned into 10-year groups: 10-19, 20-29, ..., 90-99)
   - Sex (encoded: Male=1, Female=0)
   - Marital status (one-hot encoded)
   - Education (one-hot encoded)
   - Education number (one-hot encoded)

2. **Geographic Features**:
   - Town frequency encoding (normalized frequency)

3. **Employment Features**:
   - Job title (label encoded)
   - Occupation level
   - Employment duration (years + months combined, capped at 95th percentile)
   - Employment duration (square root transformed for skew reduction)

4. **Financial Features**:
   - Salary band (parsed, converted to GBP)
   - Capital gain/loss (net profit calculated, capped at 97th percentile)
   - Net profit (square root transformed)

5. **Social Media Features**:
   - Facebook familiarity (standardized)
   - Facebook views (standardized)

6. **Categorical Features** (One-Hot Encoded):
   - Religion
   - Relationship status
   - Workclass
   - Race
   - Native country

7. **Frequency Encoded**:
   - Demographic characteristic frequency

### Excluded Features
The following columns are automatically excluded:
- Identifiers: `participant_id`, `postcode`, `company_email`, `paye`
- Name fields: `first_name`, `last_name`, `full_name`, `full_name_clean`
- Date fields: `dob`, `dob_parsed`, `age_from_dob`
- Temporary fields: `name_clean_temp`, `first_last`, `new_mortgage`
- High-cardinality categoricals (>50 unique values): `hours_per_week`

### Feature Count
- Final feature count varies based on one-hot encoding
- Approximately 100+ features after preprocessing

---

## Model Architecture

### Model 1: Logistic Regression

**Algorithm**: Logistic Regression (sklearn)

**Hyperparameters** (from notebook Cell 60):
- `C`: 0.5 (regularization strength)
- `penalty`: 'l2' (Ridge regularization)
- `solver`: 'liblinear'
- `max_iter`: 1000
- `class_weight`: 'balanced' (handles class imbalance)
- `random_state`: 1234

**Preprocessing Pipeline**:
1. Numeric features:
   - Missing value imputation (median)
   - Standard scaling
2. Categorical features:
   - Missing value imputation (most frequent)
   - One-hot encoding

### Model 2: Random Forest

**Algorithm**: Random Forest Classifier (sklearn)

**Hyperparameters** (from notebook Cell 61):
- `n_estimators`: 1000
- `class_weight`: 'balanced' (handles class imbalance)
- `random_state`: 1234
- `n_jobs`: -1 (parallel processing)

**Preprocessing Pipeline**: Same as Logistic Regression

---

## Performance

### Evaluation Metrics

#### Logistic Regression Performance (Test Set)
Based on test set of 321 samples (from notebook Cell 62):

| Metric | Class 0 (No Account) | Class 1 (Account Created) | Overall |
|--------|---------------------|--------------------------|---------|
| Precision | 0.99 | 0.50 | - |
| Recall | 0.92 | 0.85 | - |
| F1-Score | 0.95 | 0.63 | - |
| Support | 294 | 27 | 321 |
| **ROC AUC** | - | - | **0.9699** |

**Precision@K** (from notebook Cell 62):
- Precision@50: 0.48
- Precision@100: 0.27
- Precision@200: 0.135

#### Random Forest Performance (Test Set)
Based on test set of 321 samples (from notebook Cell 62):

| Metric | Class 0 (No Account) | Class 1 (Account Created) | Overall |
|--------|---------------------|--------------------------|---------|
| Precision | 0.97 | 0.94 | - |
| Recall | 1.00 | 0.63 | - |
| F1-Score | 0.98 | 0.76 | - |
| Support | 294 | 27 | 321 |
| **ROC AUC** | - | - | **0.9936** |

**Precision@K** (from notebook Cell 62):
- Precision@50: 0.54
- Precision@100: 0.27
- Precision@200: 0.135

### Model Comparison
- **Random Forest** has higher ROC AUC (0.9936 vs 0.9699)
- **Logistic Regression** has better recall for positive class (0.85 vs 0.63)
- **Random Forest** has better precision for positive class (0.94 vs 0.50)
- Both models show strong performance on majority class (Class 0)
- **Random Forest** provides better overall performance and precision
- **Logistic Regression** is better for scenarios requiring high recall (finding all potential customers)

### Threshold Optimization
- Default threshold: 0.5
- Optimal threshold can be found using `find_optimal_threshold()` function
- Threshold selection depends on business priorities (precision vs recall)

---

## Limitations

### Data Limitations
1. **Class Imbalance**: Significant class imbalance (majority class ~91%, minority class ~9%)
2. **Missing Data**: High percentage of missing target values (90.6% in campaign data)
3. **Small Training Set**: Only ~1,604 rows with labels after filtering
4. **Data Age**: [TO BE FILLED: If data is historical, may not reflect current patterns]

### Model Limitations
1. **Feature Dependencies**: Model performance depends on data quality and feature availability
2. **Temporal Validity**: Model trained on historical data may not generalize to future periods
3. **Geographic Bias**: Training data may be biased toward specific regions
4. **Interpretability**: Random Forest is less interpretable than Logistic Regression

### Known Issues
- High-cardinality categorical features are dropped (>50 unique values)
- Model may not perform well on customers with missing key features
- No explicit handling of data drift over time

---

## Ethical Considerations

### Fairness
- Model should be evaluated for fairness across:
  - Age groups
  - Gender
  - Geographic regions
  - Socioeconomic status
- Regular bias audits recommended

### Privacy
- Model uses personal information (names, demographics, financial data)
- Ensure compliance with GDPR and data protection regulations
- Customer consent required for data usage

### Transparency
- Model decisions should be explainable to stakeholders
- SHAP values available for model interpretation
- Feature importance plots provided

---

## Model Explainability

### Interpretability Methods
1. **Feature Importance**:
   - Logistic Regression: Coefficient magnitudes
   - Random Forest: Feature importances
   - Top 20 features visualized
   - **Saved plots**: `output/explainability_plots/{model_name}_feature_importance.png`

2. **SHAP Values**:
   - TreeExplainer for Random Forest
   - Explainer for Logistic Regression
   - Summary plots available
   - **Saved plots**: `output/explainability_plots/{model_name}_shap_summary.png`

3. **Model Coefficients**:
   - Logistic Regression coefficients show feature impact
   - Positive coefficients increase probability
   - Negative coefficients decrease probability

### Key Features (Top Contributors)
[TO BE FILLED: Based on feature importance analysis]
- Example: Age group, education level, employment duration, etc.

### Explainability Output Files
All explainability plots are automatically saved to `output/explainability_plots/`:
- `logistic_regression_feature_importance.png` - Top 20 features for Logistic Regression
- `random_forest_feature_importance.png` - Top 20 features for Random Forest
- `logistic_regression_shap_summary.png` - SHAP summary plot for Logistic Regression
- `random_forest_shap_summary.png` - SHAP summary plot for Random Forest

---

## Usage Instructions

### Model Loading
```python
# Models are trained via run_pipeline.py
from src.model_train import train_models
logreg, rf, (X_test, y_test) = train_models(df)
```

### Prediction
```python
# Get probabilities
probabilities = model.predict_proba(X_new)[:, 1]

# Get predictions (default threshold 0.5)
predictions = model.predict(X_new)

# Use custom threshold
from src.evaluation import evaluate_with_threshold
results = evaluate_with_threshold(model, X_new, y_new, threshold=0.4)
```

### Model Evaluation
```python
from src.evaluation import evaluate_model
evaluate_model(model, X_test, y_test, "Model Name")
```

### Explainability
```python
from src.evaluation import explain_model_shap, plot_feature_importance
explain_model_shap(model, X_test, "Model Name", model_type="rf")
plot_feature_importance(model, "Model Name", top_n=20)
```

---

## Maintenance

### Monitoring Recommendations
1. **Performance Monitoring**:
   - Track ROC AUC, precision, recall over time
   - Monitor for model drift
   - Set up alerts for significant performance degradation

2. **Data Monitoring**:
   - Monitor feature distributions for drift
   - Track missing value rates
   - Verify data quality

3. **Business Metrics**:
   - Track actual account creation rates
   - Compare predicted vs actual conversion
   - Monitor campaign ROI

### Retraining Schedule
- **Recommended**: Quarterly or when performance degrades
- **Trigger Events**:
  - Significant drop in performance metrics
  - Major changes in customer base
  - New data sources available
  - Regulatory requirements

### Model Versioning
- Version models with timestamps
- Track model performance across versions
- Maintain model artifacts (pickle files, feature lists)

---

## References

### Code Repository
- Main pipeline: `run_pipeline.py`
- Model training: `src/model_train.py`
- Feature engineering: `src/feature_engineering.py`
- Evaluation: `src/evaluation.py`

### Documentation
- Requirements coverage: `docs/requirements_coverage.md`
- EDA analysis: `run_eda.py` and `src/eda.py`
- Notebook reference: `notebooks/02_EDA_Campaign_Mortgage.ipynb`

### Dependencies
- Python packages: See `requirements.txt`
- Key libraries: scikit-learn, pandas, numpy, imbalanced-learn, shap (optional)

---

## Contact Information

**Model Maintainer**: Divya Thekke Kanapram
**Email**: divyakanapram@hotmail.com
**Last Updated**: 20-11-2025

---

## Appendix

### Model Artifacts
- Trained models: --
- Preprocessing pipeline: Included in model object
- Feature list: Generated automatically from training data

### Training Configuration
- Random seed: 1234 (consistent across train/test split and model training)
- Test size: 20%
- Stratification: Yes
- Cross-validation: Not used (single train/test split)

### Performance Benchmarks
- Baseline (majority class): [TO BE FILLED: e.g., 91.4% accuracy]
- Model improvement: [TO BE FILLED: Improvement over baseline]

