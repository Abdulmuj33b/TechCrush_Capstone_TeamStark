"""Helper functions for loading and cleaning the heart dataset.
Beginner-friendly and heavily commented."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_from_excel(path):
    """Load Excel file into a pandas DataFrame. Supports .xls/.xlsx."""
    df = pd.read_excel(path)
    return df

def basic_clean(df):
    """Simple cleaning: lowercase columns, strip whitespace, and drop exact duplicates."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df = df.drop_duplicates()
    # Replace empty strings with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df

def split_data(df, target='target', test_size=0.2, random_state=42):
    """Split features and target into train/test sets with stratification if possible."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
