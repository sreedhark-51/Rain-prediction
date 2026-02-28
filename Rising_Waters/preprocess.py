"""
Data Preprocessing Module for Flood Prediction

Handles:
- Loading dataset
- Missing value treatment
- Feature scaling
- Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(filepath='data/flood_data.csv'):
    """
    Load the flood dataset
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found!")
        raise


def handle_missing_values(data):
    """
    Handle missing values in the dataset
    
    Args:
        data (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Dataset with missing values handled
    """
    missing_count = data.isnull().sum().sum()
    
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values")
        print("Missing values per column:")
        print(data.isnull().sum())
        
        # Fill missing values with mean
        data = data.fillna(data.mean())
        print("✅ Missing values filled with mean")
    else:
        print("✅ No missing values found!")
    
    return data


def scale_features(X_train, X_test, feature_names):
    """
    Scale features using StandardScaler
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Testing features
        feature_names (list): Names of features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled using StandardScaler")
    print(f"📊 Scaling parameters learned from training data")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(filepath='data/flood_data.csv', test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath (str): Path to the CSV file
        test_size (float): Fraction of data for testing (default 0.2 = 80/20 split)
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing:
            - X_train: Training features (scaled)
            - X_test: Testing features (scaled)
            - y_train: Training labels
            - y_test: Testing labels
            - feature_names: Names of features
            - scaler: Fitted StandardScaler object
    """
    print("\n" + "="*60)
    print("🔧 PREPROCESSING PIPELINE STARTED")
    print("="*60)
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading Data")
    data = load_data(filepath)
    
    # Step 2: Handle missing values
    print("\n🧹 Step 2: Handling Missing Values")
    data = handle_missing_values(data)
    
    # Step 3: Separate features and target
    print("\n🔀 Step 3: Separating Features and Target")
    
    # Get feature names (all columns except 'flood_risk')
    feature_names = [col for col in data.columns if col != 'flood_risk']
    X = data[feature_names].values
    y = data['flood_risk'].values
    
    print(f"✅ Features: {feature_names}")
    print(f"✅ Target: flood_risk")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Step 4: Train-test split (80/20)
    print(f"\n✂️  Step 4: Train-Test Split ({int((1-test_size)*100)}/{int(test_size*100)})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Stratified split to maintain class distribution
    )
    
    print(f"✅ Training set size: {X_train.shape[0]} samples ({int((1-test_size)*100)}%)")
    print(f"✅ Testing set size: {X_test.shape[0]} samples ({int(test_size*100)}%)")
    print(f"✅ Training target distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        pct = (count / len(y_train)) * 100
        print(f"   - Class {label}: {count} samples ({pct:.1f}%)")
    
    # Step 5: Feature scaling
    print(f"\n📏 Step 5: Feature Scaling")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_names)
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60 + "\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler
    }


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessed_data = preprocess_data()
    
    print("Preprocessed Data Summary:")
    print(f"X_train shape: {preprocessed_data['X_train'].shape}")
    print(f"X_test shape: {preprocessed_data['X_test'].shape}")
    print(f"Feature names: {preprocessed_data['feature_names']}")
