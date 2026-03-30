"""
config.py
The central configuration hub for the SWaT Anomaly Detection pipeline.
"""
import os

# ==========================================
# FILE PATHS
# ==========================================
# Use absolute or relative paths based on your repo structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "SWaT_2019")

# ==========================================
# MODEL SETTINGS (Hugging Face)
# ==========================================
# We will start with Chronos-T5 (Amazon's time-series model) 
# because it's lightweight and perfect for Colab's T4 GPU.
HF_MODEL_NAME = "amazon/chronos-t5-small"

# ==========================================
# SLIDING WINDOW PARAMETERS
# ==========================================
# How many time-steps (rows) the model looks at to make a decision
WINDOW_SIZE = 60      

# How far to slide the window forward during training 
# (Smaller stride = more overlapping data = better training)
TRAIN_STRIDE = 10     

# Stride for testing (Must be equal to WINDOW_SIZE to prevent data leakage!)
TEST_STRIDE = 60      

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 32       # How many windows to process at once
LEARNING_RATE = 1e-4  # Standard starting point for fine-tuning
EPOCHS = 5            # How many times to loop through the whole dataset

# ==========================================
# SYSTEM SETTINGS
# ==========================================
# The column name we will create to flag attacks
LABEL_COLUMN = "is_attack"