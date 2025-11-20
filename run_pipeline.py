from src.load_data import load_data
from src.data_cleaning import basic_cleaning
from src.feature_engineering import engineer_features
from src.model_train import train_models
from src.evaluation import evaluate_model, explain_model_shap, plot_feature_importance


print("Loading data...")
campaign, mortgage = load_data()


print("Cleaning data...")
campaign, mortgage = basic_cleaning(campaign, mortgage)


print("Merging...")
# Merge on full_name_clean and age (as per notebook Cell 9)
merged = campaign.merge(
    mortgage, 
    left_on=['full_name_clean', 'age'],
    right_on=['full_name_clean', 'age_from_dob'],
    how='inner', 
    suffixes=('_camp', '_mort')
)

print(f"Total records after merge: {merged.shape[0]}")

# Drop temporary columns (as per notebook Cell 14)
print("Dropping temporary columns...")
columns_to_drop = [
    'participant_id', 'name_title', 'first_name', 'last_name', 'postcode',
    'company_email', 'full_name_clean', 'full_name', 'dob', 'paye',
    'name_clean_temp', 'first_last', 'dob_parsed', 'age_from_dob', 'new_mortgage'
]
columns_to_drop = [col for col in columns_to_drop if col in merged.columns]
if columns_to_drop:
    merged = merged.drop(columns=columns_to_drop)
    print(f"Dropped {len(columns_to_drop)} temporary columns")

print("Engineering features...")
df = engineer_features(merged)


print("Training models...")
logreg, rf, (X_test, y_test) = train_models(df)


print("Evaluating models...")
thresholds_to_monitor = (0.4, 0.5)
logreg_pred, logreg_prob = evaluate_model(
    logreg, X_test, y_test, "Logistic Regression", additional_thresholds=thresholds_to_monitor
)
rf_pred, rf_prob = evaluate_model(
    rf, X_test, y_test, "Random Forest", additional_thresholds=thresholds_to_monitor
)

# Threshold optimization (optional)
print("\n" + "="*50)
print("THRESHOLD OPTIMIZATION")
print("="*50)
from src.evaluation import find_optimal_threshold

print("\nFinding optimal threshold for Logistic Regression (optimizing F1)...")
logreg_threshold, logreg_metrics = find_optimal_threshold(y_test.values, logreg_prob, metric='f1')
print(f"Optimal threshold: {logreg_threshold:.3f}")
print(f"Metrics at optimal threshold: F1={logreg_metrics['f1']:.4f}, "
      f"Precision={logreg_metrics['precision']:.4f}, Recall={logreg_metrics['recall']:.4f}")

print("\nFinding optimal threshold for Random Forest (optimizing F1)...")
rf_threshold, rf_metrics = find_optimal_threshold(y_test.values, rf_prob, metric='f1')
print(f"Optimal threshold: {rf_threshold:.3f}")
print(f"Metrics at optimal threshold: F1={rf_metrics['f1']:.4f}, "
      f"Precision={rf_metrics['precision']:.4f}, Recall={rf_metrics['recall']:.4f}")

print("\n" + "="*50)
print("MODEL EXPLAINABILITY")
print("="*50)

# Feature importance plots
print("\nPlotting feature importance...")
plot_feature_importance(logreg, "Logistic Regression", top_n=20)
plot_feature_importance(rf, "Random Forest", top_n=20)

# SHAP explanations (optional, requires shap library)
print("\nGenerating SHAP explanations...")
print("Note: This requires the 'shap' library. Install with: pip install shap")
explain_model_shap(rf, X_test, "Random Forest", model_type="rf")
explain_model_shap(logreg, X_test, "Logistic Regression", model_type="logreg")