"""
Usage Examples
Demonstrate how to use the project modules in your own code
"""

# ============================================================================
# EXAMPLE 1: Load Data
# ============================================================================

from src.data_loader import DataLoader

# Load data
loader = DataLoader()
df = loader.load_data()

# Get data info
info = loader.get_data_info()
print(f"Dataset shape: {info['shape']}")
print(f"Missing values: {info['missing_values']}")


# ============================================================================
# EXAMPLE 2: Preprocess Data
# ============================================================================

from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Prepare features and target
X, y = preprocessor.prepare_features_and_target(df, target_column='diagnosis')
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Scale features
X_scaled = preprocessor.scale_features(X)
print(f"Scaled features shape: {X_scaled.shape}")


# ============================================================================
# EXAMPLE 3: Train Models
# ============================================================================

from src.model_trainer import ModelTrainer

trainer = ModelTrainer(test_size=0.2, random_state=42)

# Split data
trainer.split_data(X, y)

# Train all models
results = trainer.train_all_models()

# Get sorted results
sorted_results = trainer.get_sorted_results()
for model_name, accuracy in sorted_results:
    print(f"{model_name}: {accuracy:.4f}")

# Get best model info
best_model_info = trainer.get_best_model_info()
print(f"\nBest Model: {best_model_info['name']}")
print(f"Accuracy: {best_model_info['accuracy']:.4f}")


# ============================================================================
# EXAMPLE 4: Cross-Validation
# ============================================================================

cv_results = trainer.cross_validate_models(cv=5)
for model_name, cv_data in cv_results.items():
    print(f"{model_name}:")
    print(f"  Mean: {cv_data['mean']:.4f}")
    print(f"  Std: {cv_data['std']:.4f}")


# ============================================================================
# EXAMPLE 5: Hyperparameter Tuning
# ============================================================================

tuning_results = trainer.hyperparameter_tuning(model_name='Random Forest')
print(f"Best Parameters: {tuning_results['best_params']}")
print(f"Best CV Score: {tuning_results['best_cv_score']:.4f}")
print(f"Test Accuracy: {tuning_results['test_accuracy']:.4f}")


# ============================================================================
# EXAMPLE 6: Data Analysis
# ============================================================================

from src.data_analyzer import DataAnalyzer

analyzer = DataAnalyzer(df)

# Get basic statistics
stats = analyzer.get_basic_statistics()
print(f"Dataset shape: {stats['shape']}")

# Get diagnosis distribution
dist = analyzer.get_diagnosis_distribution()
print(f"Diagnosis distribution: {dist}")

# Calculate feature importance
importance_df = analyzer.calculate_feature_importance(X, y)
print("\nTop 5 important features:")
print(analyzer.get_top_features(5))


# ============================================================================
# EXAMPLE 7: Save and Load Model
# ============================================================================

# Save model
trainer.save_model('models/best_model.pkl')
print("Model saved!")

# Load model
trainer.load_model('models/best_model.pkl')
print("Model loaded!")


# ============================================================================
# EXAMPLE 8: Make Predictions
# ============================================================================

import pandas as pd

# Prepare input data (must have same features as training)
sample_data = X.iloc[:5]  # Get first 5 samples

# Scale the data
sample_scaled = trainer.scaler.transform(sample_data)

# Make predictions
predictions = trainer.predict(sample_scaled)
print(f"Predictions: {predictions}")


# ============================================================================
# EXAMPLE 9: Using Model Manager
# ============================================================================

from src.model_utils import ModelManager

manager = ModelManager()

# Load all artifacts
if manager.load_artifacts():
    print("✓ Models loaded successfully")
    
    # Make prediction on new data
    result = manager.predict(X.iloc[:1])
    print(f"Prediction: {result['diagnosis']}")
    print(f"Probabilities: {result['probabilities']}")


# ============================================================================
# EXAMPLE 10: Complete Pipeline in One Script
# ============================================================================

def complete_pipeline():
    """Run the complete ML pipeline"""
    
    print("=" * 60)
    print("COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    loader = DataLoader()
    df = loader.load_data()
    print(f"✓ Loaded {len(df)} samples")
    
    # 2. Analyze data
    print("\n[2] Analyzing data...")
    analyzer = DataAnalyzer(df)
    print(analyzer.get_data_info_string())
    
    # 3. Preprocess
    print("\n[3] Preprocessing data...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df)
    print(f"✓ Features shape: {X.shape}")
    
    # 4. Train models
    print("\n[4] Training models...")
    trainer = ModelTrainer()
    trainer.split_data(X, y)
    results = trainer.train_all_models()
    print("✓ All models trained")
    
    # 5. Display results
    print("\n[5] Results:")
    for name, accuracy in trainer.get_sorted_results():
        print(f"  {name}: {accuracy:.4f}")
    
    # 6. Save model
    print("\n[6] Saving model...")
    trainer.save_model('models/best_model.pkl')
    print("✓ Model saved")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED!")
    print("=" * 60)

# Uncomment to run:
# complete_pipeline()


# ============================================================================
# EXAMPLE 11: Custom Model Training
# ============================================================================

def custom_model_training():
    """Train a single model with custom parameters"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train custom model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Custom Model Accuracy: {accuracy:.4f}")

# Uncomment to run:
# custom_model_training()


if __name__ == "__main__":
    print("""
    USAGE EXAMPLES
    
    This file contains examples of how to use the project modules.
    
    Examples included:
    1. Loading data
    2. Preprocessing data
    3. Training models
    4. Cross-validation
    5. Hyperparameter tuning
    6. Data analysis
    7. Saving/loading models
    8. Making predictions
    9. Using Model Manager
    10. Complete pipeline
    11. Custom model training
    
    Uncomment any example in main section or import and use in your code.
    """)
