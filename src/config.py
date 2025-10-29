import os
from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_PATH = Path("data")
RAW_DATA_FILE = DATA_PATH/"census-bureau.data"
RAW_COLS_FILE = DATA_PATH/"census-bureau.columns"

# Model parameters
RARE_THRESH = 0.005  # 0.5% by weighted frequency
TARGET_MIN_RECALL = 0.70

# Clustering parameters
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8  # Reduced from 8 to 4 for faster execution