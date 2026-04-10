"""
Data Loading and Preprocessing Module
Handles loading data from CSV and database operations
"""

import pandas as pd
import os
from pathlib import Path


class DataLoader:
    """Handle data loading from CSV files"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the DataLoader
        
        Args:
            data_path: Path to the CSV file. If None, uses default location.
        """
        if data_path is None:
            # Get the project root and construct path
            current_dir = Path(__file__).resolve().parent.parent
            data_path = os.path.join(current_dir, 'data', 'CancerDecease.csv')
        
        self.data_path = data_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded data
        
        Returns:
            dict: Data information including shape, columns, dtypes
        """
        if self.df is None:
            self.load_data()
        
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'size': len(self.df)
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the loaded dataframe
        
        Returns:
            pandas.DataFrame: The dataframe
        """
        if self.df is None:
            self.load_data()
        return self.df
