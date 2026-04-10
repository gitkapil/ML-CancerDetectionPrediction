"""
Exploratory Data Analysis Module
Handles data visualization and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class DataAnalyzer:
    """Perform exploratory data analysis on the dataset"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataAnalyzer
        
        Args:
            df: Input dataframe
        """
        self.df = df
        self.feature_importance = None
    
    def get_basic_statistics(self) -> dict:
        """
        Get basic statistical information
        
        Returns:
            dict: Statistical information
        """
        stats = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'describe': self.df.describe().to_dict(),
            'correlation': self.df.corr().to_dict()
        }
        return stats
    
    def get_diagnosis_distribution(self) -> dict:
        """
        Get diagnosis distribution
        
        Returns:
            dict: Distribution counts
        """
        if 'diagnosis' in self.df.columns:
            dist = self.df['diagnosis'].value_counts().to_dict()
            return dist
        return {}
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest
        
        Args:
            X: Features
            y: Target
        
        Returns:
            pandas.DataFrame: Feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N important features
        
        Args:
            n: Number of top features
        
        Returns:
            pandas.DataFrame: Top features
        """
        if self.feature_importance is not None:
            return self.feature_importance.head(n)
        return pd.DataFrame()
    
    def get_data_info_string(self) -> str:
        """
        Get formatted data information
        
        Returns:
            str: Formatted information
        """
        buffer = []
        buffer.append("=" * 60)
        buffer.append("DATA INFORMATION")
        buffer.append("=" * 60)
        buffer.append(f"Shape: {self.df.shape}")
        buffer.append(f"Columns: {len(self.df.columns)}")
        buffer.append(f"Rows: {len(self.df)}")
        buffer.append(f"\nMissing Values:\n{self.df.isnull().sum()}")
        buffer.append(f"\nData Types:\n{self.df.dtypes}")
        return "\n".join(buffer)
    
    def get_summary_statistics(self) -> str:
        """
        Get summary statistics as string
        
        Returns:
            str: Formatted statistics
        """
        return str(self.df.describe())
