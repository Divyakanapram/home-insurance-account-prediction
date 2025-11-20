# Unit Tests

This directory contains comprehensive unit tests for the insurance account creation prediction pipeline.

## Test Structure

- `test_load_data.py` - Tests for data loading functionality
- `test_data_cleaning.py` - Tests for data cleaning and preprocessing
- `test_feature_engineering.py` - Tests for feature engineering transformations
- `test_model_train.py` - Tests for model training pipeline
- `test_evaluation.py` - Tests for model evaluation and explainability
- `test_integration.py` - Integration tests for the full pipeline
- `conftest.py` - Shared pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_load_data.py
```

### Run specific test class
```bash
pytest tests/test_feature_engineering.py::TestEngineerFeatures
```

### Run specific test function
```bash
pytest tests/test_feature_engineering.py::TestEngineerFeatures::test_target_variable_cleaning
```

### Run with verbose output
```bash
pytest -v
```

### Run only fast tests (exclude slow/integration tests)
```bash
pytest -m "not slow and not integration"
```

## Test Coverage

The tests aim for comprehensive coverage of:
- ✅ All public functions
- ✅ Edge cases (empty data, missing values, etc.)
- ✅ Error handling
- ✅ Data type validation
- ✅ Pipeline integration

## Fixtures

Shared test fixtures are defined in `conftest.py`:
- `sample_campaign_data` - Sample campaign dataset
- `sample_mortgage_data` - Sample mortgage dataset
- `sample_merged_data` - Sample merged dataset
- `sample_features_data` - Sample feature-engineered dataset
- `temp_data_dir` - Temporary directory with test CSV files

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **Test naming**: Use descriptive names starting with `test_`
2. **Test organization**: Group related tests in classes
3. **Use fixtures**: Reuse existing fixtures or create new ones in `conftest.py`
4. **Test edge cases**: Include tests for empty data, missing values, etc.
5. **Assertions**: Use clear, specific assertions
6. **Documentation**: Add docstrings explaining what each test validates

Example:
```python
def test_new_functionality_basic_case(sample_data):
    """Test that new functionality works with valid input."""
    result = new_function(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
```

## Continuous Integration

Tests should pass before merging code. The test suite is designed to:
- Run quickly (< 1 minute for all tests)
- Be deterministic (same results every run)
- Not require external services
- Use mock data when possible

