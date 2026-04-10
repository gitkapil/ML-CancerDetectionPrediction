"""
Model Training Module
Handles training multiple ML models and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import all models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import joblib
from pathlib import Path


class ModelTrainer:
    """Train and evaluate multiple machine learning models"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            test_size: Proportion of test set
            random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.models = self._create_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def _create_models(self) -> dict:
        """
        Create all models
        
        Returns:
            dict: Dictionary of models
        """
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gaussian Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'AdaBoost': AdaBoostClassifier(random_state=self.random_state)
        }
        return models
    
    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_all_models(self) -> dict:
        """
        Train all models and evaluate
        
        Returns:
            dict: Results for all models
        """
        if self.X_train_scaled is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        for name, model in self.models.items():
            # Train
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predict
            predictions = model.predict(self.X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, predictions)
            self.results[name] = {
                'accuracy': accuracy,
                'model': model,
                'predictions': predictions
            }
        
        # Find best model
        best_result = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        self.best_model_name = best_result[0]
        self.best_model = best_result[1]['model']
        
        return self.results
    
    def get_sorted_results(self) -> list:
        """
        Get results sorted by accuracy
        
        Returns:
            list: Sorted results (name, accuracy) tuples
        """
        sorted_results = sorted(
            [(name, data['accuracy']) for name, data in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_results
    
    def get_best_model_info(self) -> dict:
        """
        Get information about the best model
        
        Returns:
            dict: Best model information
        """
        return {
            'name': self.best_model_name,
            'accuracy': self.results[self.best_model_name]['accuracy'],
            'model': self.best_model
        }
    
    def cross_validate_models(self, cv: int = 5) -> dict:
        """
        Perform cross-validation on all models
        
        Args:
            cv: Number of folds
        
        Returns:
            dict: Cross-validation results
        """
        cv_results = {}
        
        for name, model in self.models.items():
            cv_scores = cross_val_score(
                model, 
                self.X_train_scaled, 
                self.y_train, 
                cv=cv
            )
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str = 'Random Forest') -> dict:
        """
        Perform hyperparameter tuning on best model
        
        Args:
            model_name: Name of model to tune
        
        Returns:
            dict: Tuning results
        """
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Tuning not implemented for {model_name}")
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1
        )
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        best_model = grid_search.best_estimator_
        test_accuracy = accuracy_score(self.y_test, best_model.predict(self.X_test_scaled))
        
        return {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': test_accuracy,
            'model': best_model
        }
    
    def get_classification_report(self, model_name: str = None) -> str:
        """
        Get classification report for a model
        
        Args:
            model_name: Name of model (default: best model)
        
        Returns:
            str: Classification report
        """
        if model_name is None:
            model_name = self.best_model_name
        
        predictions = self.results[model_name]['predictions']
        return classification_report(self.y_test, predictions)
    
    def get_confusion_matrix(self, model_name: str = None) -> np.ndarray:
        """
        Get confusion matrix for a model
        
        Args:
            model_name: Name of model (default: best model)
        
        Returns:
            numpy.ndarray: Confusion matrix
        """
        if model_name is None:
            model_name = self.best_model_name
        
        predictions = self.results[model_name]['predictions']
        return confusion_matrix(self.y_test, predictions)
    
    def save_model(self, model_path: str):
        """
        Save the best model
        
        Args:
            model_path: Path to save the model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, model_path)
    
    def load_model(self, model_path: str):
        """
        Load a trained model
        
        Args:
            model_path: Path to the model file
        """
        self.best_model = joblib.load(model_path)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the best model
        
        Args:
            X: Input features
        
        Returns:
            numpy.ndarray: Predictions
        """
        if self.best_model is None:
            raise ValueError("No model available. Train a model first.")
        
        return self.best_model.predict(X)
