import os

# Base directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

CAMPAIGN_FILE = os.path.join(DATA_DIR, "campaign.csv")
MORTGAGE_FILE = os.path.join(DATA_DIR, "mortgage.csv")
