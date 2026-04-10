# Cancer Diagnosis Prediction System

## 🎯 Overview

A professional machine learning project for predicting cancer diagnosis using various classification algorithms. This project demonstrates best practices in data science workflows including data preprocessing, model training, evaluation, and interactive visualization through a Streamlit dashboard.

## ✨ Features

- **Data Exploration**: Comprehensive exploratory data analysis with visualizations
- **Multiple Models**: Compare 8 different ML algorithms for optimal performance
- **Model Optimization**: Hyperparameter tuning for best model accuracy
- **Cross-Validation**: Robust model evaluation using k-fold cross-validation
- **Feature Importance**: Identify the most important features for predictions
- **Interactive Dashboard**: Streamlit app for real-time predictions and analysis
- **Production-Ready**: Well-organized, modular code structure following best practices

## 📦 Project Structure

```
ML-CancelDeceasePrediction/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing and scaling
│   ├── model_trainer.py         # Model training and evaluation
│   └── data_analyzer.py         # Statistical analysis and visualization
├── data/                        # Data directory
│   └── CancerDecease.csv       # Input dataset
├── models/                      # Trained models and artifacts
│   ├── best_model.pkl          # Trained model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_names.pkl       # Feature names
├── notebooks/                   # Jupyter notebooks (optional)
├── app.py                       # Streamlit dashboard application
├── run_pipeline.py             # ML pipeline execution script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd ML-CancelDeceasePrediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the data file is in the correct location:**
   ```bash
   # Copy or move CancerDecease.csv to data/ directory
   mv CancerDecease.csv data/
   ```

### Running the Pipeline

To train the models and execute the complete ML pipeline:

```bash
python run_pipeline.py
```

This will:
- Load and analyze the dataset
- Preprocess the data
- Train all 8 models
- Evaluate model performance
- Save trained models and artifacts

### Launching the Dashboard

To start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## 📊 Dashboard Features

### 📈 Data Overview
- Dataset statistics and visualizations
- Diagnosis distribution charts
- Data sample preview

### 🔬 Exploratory Analysis
- Feature distributions and statistics
- Correlation heatmap
- Interactive feature selection

### 🎯 Model Performance
- Model accuracy comparison
- Cross-validation results
- Detailed performance metrics

### ⭐ Feature Importance
- Top N important features
- Feature contribution visualization
- Impact on diagnosis prediction

### 🔮 Make Predictions
- Interactive input form for patient data
- Real-time prediction results
- Diagnosis classification (Healthy/Cancer)

## 🤖 Machine Learning Models

The project trains and compares the following models:

1. **Random Forest** - Ensemble method with decision trees
2. **Logistic Regression** - Linear classification model
3. **Decision Tree** - Single decision tree classifier
4. **Support Vector Machine (SVM)** - Kernel-based classifier
5. **K-Nearest Neighbors** - Instance-based learning
6. **Gaussian Naive Bayes** - Probabilistic classifier
7. **Gradient Boosting** - Ensemble gradient boosting
8. **AdaBoost** - Adaptive boosting classifier

## 📝 Code Organization

### `src/data_loader.py`
Handles loading data from CSV files with error handling and path management.

```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data()
```

### `src/preprocessing.py`
Preprocesses data including feature scaling and target encoding.

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X, y = preprocessor.prepare_features_and_target(df)
X_scaled = preprocessor.scale_features(X)
```

### `src/model_trainer.py`
Trains multiple models and handles evaluation.

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.split_data(X, y)
results = trainer.train_all_models()
```

### `src/data_analyzer.py`
Performs exploratory data analysis and feature importance calculation.

```python
from src.data_analyzer import DataAnalyzer

analyzer = DataAnalyzer(df)
importance = analyzer.calculate_feature_importance(X, y)
```

## 🔧 Configuration

Default configuration values can be modified:

- **Test Size**: 0.2 (80/20 split)
- **Random State**: 42 (reproducibility)
- **Cross-Validation Folds**: 5
- **Feature Scaling**: StandardScaler

## 📈 Expected Performance

The Random Forest model typically achieves >95% accuracy on this dataset.

Performance varies based on:
- Data quality and preprocessing
- Model hyperparameters
- Train-test split
- Cross-validation results

## 🛠️ Troubleshooting

### Port Already in Use
If port 8501 is already in use, run:
```bash
streamlit run app.py --server.port 8502
```

### Models Not Training
Ensure the data file is in the correct location:
```bash
ls -la data/CancerDecease.csv
```

### Dependency Issues
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization
- **streamlit**: Web app framework
- **joblib**: Model serialization
- **plotly**: Interactive charts

## 🎓 Learning Resources

This project demonstrates:
- Data preprocessing and feature scaling
- Model selection and comparison
- Cross-validation and hyperparameter tuning
- Feature importance analysis
- Web app development with Streamlit
- Best practices in project organization

## 💡 Future Enhancements

- [ ] Add deep learning models (Neural Networks)
- [ ] Implement SHAP for model explainability
- [ ] Add data upload functionality
- [ ] Deploy to cloud platforms (AWS, Azure, GCP)
- [ ] Add ROC curves and AUC metrics
- [ ] Implement ensemble methods
- [ ] Add batch prediction functionality

