"""
Main pipeline script
Execute the complete ML pipeline: data loading, preprocessing, training, and evaluation
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from data_analyzer import DataAnalyzer


def main():
    """
    Execute the complete ML pipeline
    """
    print("=" * 80)
    print("CANCER DECEASE PREDICTION - ML PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/5] Loading data...")
    data_loader = DataLoader()
    df = data_loader.load_data()
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    
    # 2. Data Analysis
    print("\n[2/5] Performing data analysis...")
    analyzer = DataAnalyzer(df)
    print(analyzer.get_data_info_string())
    print("\nDiagnosis Distribution:")
    print(analyzer.get_diagnosis_distribution())
    
    # 3. Data Preprocessing
    print("\n[3/5] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df, 'diagnosis')
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # 4. Model Training
    print("\n[4/5] Training models...")
    trainer = ModelTrainer()
    trainer.split_data(X, y)
    print("✓ Data split: Train/Test = 80/20")
    
    results = trainer.train_all_models()
    print("\nModel Training Results:")
    for model_name, accuracy in trainer.get_sorted_results():
        print(f"  {model_name}: {accuracy:.4f}")
    
    best_info = trainer.get_best_model_info()
    print(f"\n🏆 Best Model: {best_info['name']} (Accuracy: {best_info['accuracy']:.4f})")
    
    # 5. Cross Validation
    print("\n[5/5] Performing cross-validation...")
    cv_results = trainer.cross_validate_models()
    print("Cross-Validation Results (5-fold):")
    for model_name, cv_data in cv_results.items():
        print(f"  {model_name}: {cv_data['mean']:.4f} (+/- {cv_data['std']:.4f})")
    
    # 6. Feature Importance
    print("\n[6/6] Calculating feature importance...")
    importance_df = analyzer.calculate_feature_importance(X, y)
    print("\nTop 10 Important Features:")
    print(analyzer.get_top_features(10).to_string())
    
    # 7. Save Model
    print("\n[7/7] Saving model...")
    model_path = Path(__file__).parent / 'models' / 'best_model.pkl'
    trainer.save_model(str(model_path))
    print(f"✓ Model saved to {model_path}")
    
    # 8. Save Feature Names
    features_path = Path(__file__).parent / 'models' / 'feature_names.pkl'
    import joblib
    joblib.dump(preprocessor.get_feature_names(), str(features_path))
    print(f"✓ Feature names saved to {features_path}")
    
    # 9. Save Scaler
    scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
    joblib.dump(trainer.scaler, str(scaler_path))
    print(f"✓ Scaler saved to {scaler_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
