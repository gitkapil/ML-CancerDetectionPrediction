"""
Data Preprocessing Module
Handles data cleaning and feature preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """Handle data preprocessing and transformation"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X = None
        self.y = None
    
    def prepare_features_and_target(self, df: pd.DataFrame, target_column: str = 'diagnosis'):
        """
        Separate features and target variable
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
        
        Returns:
            tuple: (X, y) features and target
        """
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['id', 'Unnamed: 32', target_column]
        self.X = df_copy.drop(
            columns=[col for col in columns_to_drop if col in df_copy.columns and col != target_column],
            errors='ignore'
        )
        
        self.y = df_copy[target_column]
        
        return self.X, self.y
    
    def scale_features(self, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Testing features
        
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if X_train is None:
            X_train = self.X
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def encode_target(self, y):
        """
        Encode target variable
        
        Args:
            y: Target variable
        
        Returns:
            numpy.ndarray: Encoded target
        """
        return self.label_encoder.fit_transform(y)
    
    def get_feature_names(self) -> list:
        """
        Get feature names
        
        Returns:
            list: Feature column names
        """
        if self.X is not None:
            return list(self.X.columns)
        return []
