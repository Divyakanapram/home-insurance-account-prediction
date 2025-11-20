# Data Quality Analysis Report

Based on the exploratory data analysis performed in `02_EDA_Campaign_Mortgage.ipynb`, here are the answers to the data quality questions:

## 1. How Good is the Data? (Data Quality Assessment)

### Overall Data Quality: **GOOD with Some Issues**

The data quality is generally good, but there are several issues that need to be addressed:

#### **Strengths:**
- **No duplicate rows**: Both datasets are clean of duplicate records
  - Campaign dataset: 0 duplicates
  - Mortgage dataset: 0 duplicates
- **Complete records in most columns**: Most columns have no missing values
- **Consistent data types**: Data is stored consistently (all as strings initially, which is appropriate for the data format)
- **Successful data merge**: 16,591 records successfully merged from 23,621 potential name matches

#### **Data Dimensions:**
- **Campaign dataset**: 32,060 rows × 16 columns
- **Mortgage dataset**: 32,561 rows × 18 columns
- **Merged dataset**: 16,591 rows × 39 columns (after merge, before cleaning)

---

## 2. Missing Values, Different Units, and Repeated Columns

### **Missing Values:**

#### **Campaign Dataset:**
- **`created_account`**: 29,033 missing values (90.33% of records)
  - This is expected as this is the target variable we need to predict
  - Only 3,027 records have known outcomes (1,468 "no", 136 "yes")
- **`name_title`**: 12,201 missing values (38.1% of records)
  - This is acceptable as title is optional information
- **All other columns**: No missing values

#### **Mortgage Dataset:**
- **No missing values** in any column
- All 18 columns are complete

#### **Merged Dataset:**
- After merging, the only missing values are in `created_account` (90.33%)
- All other merged columns are complete

### **Different Units:**

Yes, there are different units that need standardization:

1. **Salary (`salary_band`)**:
   - Contains values in different formats: GBP, MUR (Mauritian Rupee)
   - Contains different time periods: monthly, weekly, annual
   - **Solution Applied**: Converted all to annual GBP using conversion rates:
     - GBP: 1.0
     - MUR: 0.018
   - Created `salary_value_gbp` column with standardized values

2. **Employment Duration**:
   - Stored as separate `years_with_employer` and `months_with_employer` columns
   - **Solution Applied**: Combined into `employment_duration_years` (years + months/12)

3. **Capital Gain/Loss**:
   - Stored as separate `capital_gain` and `capital_loss` columns
   - **Solution Applied**: Combined into `net_profit` (capital_gain - capital_loss)

4. **Age**:
   - Campaign: Direct age values
   - Mortgage: Date of birth (DOB) that needs to be converted to age
   - **Solution Applied**: Calculated `age_from_dob` from DOB using reference year 2018

### **Repeated Columns:**

After merging, there are some redundant columns that were created during preprocessing:

1. **Name-related columns** (dropped after merge):
   - `participant_id`, `name_title`, `first_name`, `last_name`
   - `full_name_clean`, `full_name`, `name_clean_temp`, `first_last`
   - These were used for merging but not needed in final dataset

2. **Age-related columns** (dropped after merge):
   - `dob`, `dob_parsed`, `age_from_dob`
   - Original DOB columns not needed after age calculation

3. **Other redundant columns** (dropped):
   - `postcode`, `company_email`, `paye`, `new_mortgage`

4. **Temporary encoding columns**:
   - Various one-hot encoded columns replace original categorical columns
   - Original columns like `education`, `age_group`, `marital_status` are replaced with encoded versions

---

## 3. How Can We Join the Different Datasets?

### **Join Strategy: Composite Key Matching**

The datasets are joined using a **composite key** approach with name matching and age validation:

#### **Step 1: Name Standardization**

**Campaign Dataset:**
```python
campaign["full_name_clean"] = (
    campaign["first_name"].str.strip().str.lower() + " " +
    campaign["last_name"].str.strip().str.lower()
)
```

**Mortgage Dataset:**
- Remove titles (Mr, Mrs, Ms, Miss, Dr, Prof, Sir, Madam)
- Normalize whitespace
- Extract first and last name
- Create `full_name_clean` in same format

**Result**: 23,621 potential name matches found

#### **Step 2: Age Calculation from DOB**

**Mortgage Dataset:**
```python
end_year = 2018
mortgage["dob_parsed"] = pd.to_datetime(mortgage["dob"], errors="coerce", dayfirst=True)
mortgage["age_from_dob"] = end_year - mortgage["dob_parsed"].dt.year
```

#### **Step 3: Merge on Composite Key**

**Merge Logic:**
```python
merged_df = pd.merge(
    campaign,
    mortgage,
    left_on=["full_name_clean", "age"],
    right_on=["full_name_clean", "age_from_dob"],
    how="inner",
    suffixes=("_camp", "_mort")
)
```

**Merge Results:**
- **Input**: 32,060 campaign records + 32,561 mortgage records
- **Potential matches**: 23,621 (by name only)
- **Final merged records**: 16,591 (after name + age matching)
- **Match rate**: ~51.7% of campaign records successfully matched

#### **Why This Approach Works:**

1. **Name matching alone is insufficient**: 
   - 23,621 name matches found
   - But only 16,591 match when age is also considered
   - This prevents false matches (e.g., John Smith age 30 vs John Smith age 50)

2. **Age validation adds precision**:
   - Ensures we're matching the same person
   - Reduces false positive matches significantly

3. **Inner join strategy**:
   - Only keeps records that match in both datasets
   - Ensures data completeness for matched records

#### **Challenges and Considerations:**

1. **Name variations**: 
   - Handled by lowercasing and stripping whitespace
   - Title removal handles common prefixes

2. **Age discrepancies**:
   - Age from campaign vs. calculated age from DOB may differ slightly
   - Exact match required, so some valid matches may be missed

3. **Missing matches**:
   - ~15,469 campaign records (48.3%) don't have matching mortgage records
   - Could be due to:
     - Different name formats
     - Age calculation differences
     - Records not present in mortgage dataset

#### **Alternative Join Strategies (if needed):**

1. **Fuzzy name matching**: Use Levenshtein distance for name similarity
2. **Age tolerance**: Allow ±1 year difference in age matching
3. **Multiple join attempts**: Try different name formats (with/without middle names)
4. **Left join**: Keep all campaign records, mark unmatched ones

---

## Summary

### **Data Quality Score Calculation**

The data quality score is automatically calculated using the `calculate_data_quality_score()` function in `01_exploratory_analysis.ipynb` (Cell 34). The function evaluates six key dimensions:

1. **Duplicates** (Max 2.0 points): Checks for duplicate rows in both datasets
2. **Missing Values** (Max 2.5 points): Evaluates missing value percentages
3. **Merge Success** (Max 2.0 points): Measures successful record matching rate
4. **Completeness** (Max 1.5 points): Assesses column completeness in merged dataset
5. **Consistency** (Max 1.0 points): Checks data type consistency
6. **Volume** (Max 1.0 points): Evaluates data volume sufficiency

**Total Maximum: 10.0 points**

### **Data Quality Score: Calculated Automatically**

*Note: Run Cell 34 in `01_exploratory_analysis.ipynb` to get the current calculated score based on your data.*

**Strengths:**
- ✅ No duplicate records
- ✅ Most columns complete (except target variable)
- ✅ Successful merge of 51.7% of records
- ✅ Units standardized in preprocessing

**Areas for Improvement:**
- ⚠️ High missing rate in target variable (expected for prediction task)
- ⚠️ Some missing name titles (38.1% - acceptable)
- ⚠️ Only 51.7% merge success rate (could be improved with fuzzy matching)
- ⚠️ Need for unit standardization (handled in preprocessing)

**Recommendations:**
1. Consider fuzzy name matching to improve merge rate
2. Document unit conversions for future reference
3. Validate age calculations match between datasets
4. Consider keeping unmatched records for analysis if needed

