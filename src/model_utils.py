"""
Model Utilities Module
Helper functions for model management and predictions
"""

import joblib
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ModelManager:
    """Manage model loading, saving, and predictions"""
    
    def __init__(self):
        """Initialize ModelManager"""
        self.models_dir = Path(__file__).parent / 'models'
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def load_artifacts(self):
        """Load all model artifacts"""
        model_path = self.models_dir / 'best_model.pkl'
        scaler_path = self.models_dir / 'scaler.pkl'
        features_path = self.models_dir / 'feature_names.pkl'
        
        if model_path.exists():
            self.model = joblib.load(str(model_path))
        
        if scaler_path.exists():
            self.scaler = joblib.load(str(scaler_path))
        
        if features_path.exists():
            self.feature_names = joblib.load(str(features_path))
        
        return self.model is not None
    
    def predict(self, data: pd.DataFrame) -> dict:
        """
        Make predictions on input data
        
        Args:
            data: Input dataframe
        
        Returns:
            dict: Prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Models not loaded. Call load_artifacts() first.")
        
        # Ensure columns are in correct order
        data = data[self.feature_names]
        
        # Scale features
        data_scaled = self.scaler.transform(data)
        
        # Make prediction
        prediction = self.model.predict(data_scaled)[0]
        
        # Get prediction probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(data_scaled)[0]
        else:
            probabilities = None
        
        return {
            'prediction': prediction,
            'diagnosis': 'Cancer Detected' if prediction == 1 else 'Healthy',
            'probabilities': probabilities
        }
