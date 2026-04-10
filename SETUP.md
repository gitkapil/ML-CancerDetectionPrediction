# Setup and Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for project and dependencies

## Installation Steps

### Step 1: Clone or Navigate to Project

```bash
cd ML-CancelDeceasePrediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `streamlit` - Web app framework
- `joblib` - Model serialization
- `plotly` - Interactive charts

### Step 4: Verify Data File

Ensure `CancerDecease.csv` is in the `data/` directory:

```bash
ls -la data/CancerDecease.csv
```

## Usage

### Option 1: Quick Start (Recommended)

Run the entire pipeline and launch the dashboard:

```bash
./quickstart.sh
```

Or on Windows:
```bash
python run_pipeline.py && streamlit run app.py
```

### Option 2: Manual Steps

**Step 1: Train the models**
```bash
python run_pipeline.py
```

This will:
- Load the dataset
- Preprocess the data
- Train 8 different models
- Save trained models to `models/` directory
- Display model performance comparison

**Step 2: Launch the Streamlit dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Option 3: Use in Python Code

```python
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.data_analyzer import DataAnalyzer

# Load data
loader = DataLoader()
df = loader.load_data()

# Analyze data
analyzer = DataAnalyzer(df)
print(analyzer.get_data_info_string())

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.prepare_features_and_target(df)

# Train models
trainer = ModelTrainer()
trainer.split_data(X, y)
results = trainer.train_all_models()

# Make predictions
predictions = trainer.predict(X[:5])
```

## Troubleshooting

### Issue: Python command not found
**Solution**: Ensure Python is installed and added to PATH
```bash
python --version  # or python3 --version
```

### Issue: Module not found errors
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Permission denied for quickstart.sh
**Solution**: Make it executable
```bash
chmod +x quickstart.sh
./quickstart.sh
```

### Issue: Port 8501 already in use
**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Models not found after running app
**Solution**: First run the pipeline
```bash
python run_pipeline.py
```

## Project Structure After Setup

```
ML-CancelDeceasePrediction/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   ├── data_analyzer.py
│   └── model_utils.py
├── data/
│   └── CancerDecease.csv
├── models/                    # Created after running pipeline
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── app.py                     # Streamlit dashboard
├── run_pipeline.py           # ML pipeline script
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── SETUP.md                  # This file
├── .gitignore
└── quickstart.sh            # Quick start script
```

## Deactivating Virtual Environment

When done, deactivate the virtual environment:

```bash
deactivate
```
