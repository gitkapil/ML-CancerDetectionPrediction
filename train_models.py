#!/usr/bin/env python3
"""
Standalone model training script
Trains and saves all models without launching Streamlit
"""

import pandas as pd
import sys
import os
from pathlib import Path
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from data_analyzer import DataAnalyzer


def main():
    """Execute the complete ML pipeline and save models"""
    print("=" * 80)
    print("CANCER DECEASE PREDICTION - MODEL TRAINING")
    print("=" * 80)
    
    try:
        # 1. Load Data
        print("\n[1/5] Loading data...")
        data_loader = DataLoader()
        df = data_loader.load_data()
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
        
        # 2. Data Analysis
        print("\n[2/5] Analyzing data...")
        analyzer = DataAnalyzer(df)
        print(analyzer.get_data_info_string())
        
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
        
        # 5. Save Models
        print("\n[5/5] Saving models...")
        
        # Create models directory if it doesn't exist
        models_dir = Path(__file__).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save best model
        model_path = models_dir / 'best_model.pkl'
        trainer.save_model(str(model_path))
        print(f"✓ Model saved to {model_path}")
        
        # Save feature names
        features_path = models_dir / 'feature_names.pkl'
        joblib.dump(preprocessor.get_feature_names(), str(features_path))
        print(f"✓ Feature names saved to {features_path}")
        
        # Save scaler
        scaler_path = models_dir / 'scaler.pkl'
        joblib.dump(trainer.scaler, str(scaler_path))
        print(f"✓ Scaler saved to {scaler_path}")
        
        # Verify files
        print("\n✓ Verifying saved files...")
        for file_path in [model_path, features_path, scaler_path]:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✓ {file_path.name} ({size} bytes)")
            else:
                print(f"  ✗ {file_path.name} - NOT FOUND")
        
        print("\n" + "=" * 80)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nModels are ready for predictions!")
        print("Run the Streamlit app with: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
