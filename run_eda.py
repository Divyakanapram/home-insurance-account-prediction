"""
Standalone script to run Exploratory Data Analysis on Campaign and Mortgage datasets.
"""

from src.eda import run_full_eda

if __name__ == "__main__":
    print("="*70)
    print("EXPLORATORY DATA ANALYSIS - Campaign & Mortgage Datasets")
    print("="*70)
    
    results = run_full_eda(save_plots=True)
    
    print("\n✓ EDA completed successfully!")
    print("  Check the 'output/eda_plots' directory for visualization files.")
