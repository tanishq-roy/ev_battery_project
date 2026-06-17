import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(csv_path="data/dataset.csv", target_col="eol_reached"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please upload ev_car_dataset.csv.")
        
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
        
    # Prevent Data Leakage
    leakage_columns = [
        target_col, 
        'remaining_useful_cycles', 
        'remaining_useful_life_km', 
        'remaining_useful_life_months', 
        'total_lifespan_cycles'
    ]
    cols_to_drop = [col for col in leakage_columns if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    is_classification = y.nunique() < 20 or y.dtype == 'object'
    
    stratify_param = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test, is_classification

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, is_clf = load_and_preprocess_data()
    print(f"✅ Success! Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
