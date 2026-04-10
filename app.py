"""
Streamlit Dashboard Application
Interactive dashboard for cancer diagnosis prediction and model analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from data_analyzer import DataAnalyzer
from model_trainer import ModelTrainer

# Page configuration
st.set_page_config(
    page_title="Cancer Diagnosis Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_data_and_models():
    """Load data and trained models"""
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Load model and related files
    model_path = Path(__file__).parent / 'models' / 'best_model.pkl'
    scaler_path = Path(__file__).parent / 'models' / 'scaler.pkl'
    feature_names_path = Path(__file__).parent / 'models' / 'feature_names.pkl'
    
    model = joblib.load(str(model_path)) if model_path.exists() else None
    scaler = joblib.load(str(scaler_path)) if scaler_path.exists() else None
    feature_names = joblib.load(str(feature_names_path)) if feature_names_path.exists() else None
    
    return df, model, scaler, feature_names


def header_section():
    """Display header section"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏥 Cancer Diagnosis Prediction System")
        st.markdown("*An advanced machine learning solution for cancer diagnosis prediction*")
    with col2:
        st.markdown("### 📊 Version 1.0")


def load_and_train_if_needed(df):
    """Check if models exist, if not train them"""
    model_path = Path(__file__).parent / 'models' / 'best_model.pkl'
    
    if not model_path.exists():
        st.warning("⚠️ Models not found. Training models...")
        with st.spinner("Training models... This may take a minute..."):
            # Prepare data
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_features_and_target(df, 'diagnosis')
            
            # Train models
            trainer = ModelTrainer()
            trainer.split_data(X, y)
            trainer.train_all_models()
            
            # Save models
            Path(__file__).parent.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(model_path))
            joblib.dump(trainer.scaler, str(Path(__file__).parent / 'models' / 'scaler.pkl'))
            joblib.dump(preprocessor.get_feature_names(), str(Path(__file__).parent / 'models' / 'feature_names.pkl'))
        
        st.success("✅ Models trained and saved successfully!")


def data_overview_tab(df):
    """Data Overview Tab"""
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Total Features", len(df.columns) - 1)
    with col3:
        diagnosis_0 = (df['diagnosis'] == 0).sum()
        st.metric("Healthy (0)", diagnosis_0)
    with col4:
        diagnosis_1 = (df['diagnosis'] == 1).sum()
        st.metric("Cancer (1)", diagnosis_1)
    
    st.divider()
    
    # Display raw data
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Diagnosis distribution
    st.subheader("🔍 Diagnosis Distribution")
    diagnosis_counts = df['diagnosis'].value_counts()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        diagnosis_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')
        ax.set_title('Diagnosis Distribution')
        ax.set_xticklabels(['Healthy (0)', 'Cancer (1)'], rotation=0)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        diagnosis_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        ax.set_ylabel('')
        ax.set_title('Diagnosis Percentage')
        st.pyplot(fig)


def exploratory_analysis_tab(df):
    """Exploratory Analysis Tab"""
    st.header("🔬 Exploratory Data Analysis")
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    feature_stats = df.describe().T
    st.dataframe(feature_stats, use_container_width=True)
    
    st.divider()
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    st.divider()
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    selected_features = st.multiselect(
        "Select features to visualize:",
        df.columns[:-1],
        default=df.columns[1:4].tolist()
    )
    
    if selected_features:
        cols = st.columns(len(selected_features))
        for idx, feature in enumerate(selected_features):
            with cols[idx]:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(df[feature], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{feature} Distribution')
                st.pyplot(fig)


def model_performance_tab(df):
    """Model Performance Tab"""
    st.header("🎯 Model Performance Analysis")
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df, 'diagnosis')
    
    # Train models with caching to avoid retraining
    @st.cache_resource
    def get_trained_models(_X, _y):
        trainer = ModelTrainer()
        trainer.split_data(_X, _y)
        trainer.train_all_models()
        return trainer
    
    trainer = get_trained_models(X, y)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    best_info = trainer.get_best_model_info()
    with col1:
        st.metric("🏆 Best Model", best_info['name'])
    with col2:
        st.metric("🎯 Best Accuracy", f"{best_info['accuracy']:.4f}")
    with col3:
        st.metric("📊 Total Models", len(trainer.results))
    
    st.divider()
    
    # Model comparison
    st.subheader("📊 Model Accuracy Comparison")
    sorted_results = trainer.get_sorted_results()
    results_df = pd.DataFrame(sorted_results, columns=['Model', 'Accuracy'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(results_df['Model'], results_df['Accuracy'], color='steelblue')
    ax.set_xlabel('Accuracy Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xlim([results_df['Accuracy'].min() - 0.01, 1.0])
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    st.pyplot(fig)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Cross-validation results
    st.subheader("🔄 Cross-Validation Results (5-Fold)")
    cv_results = trainer.cross_validate_models()
    cv_df = pd.DataFrame([
        {
            'Model': name,
            'Mean CV Score': data['mean'],
            'Std Dev': data['std']
        }
        for name, data in cv_results.items()
    ]).sort_values('Mean CV Score', ascending=False)
    
    st.dataframe(cv_df, use_container_width=True, hide_index=True)


def feature_importance_tab(df):
    """Feature Importance Tab"""
    st.header("⭐ Feature Importance Analysis")
    
    # Prepare data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features_and_target(df, 'diagnosis')
    
    analyzer = DataAnalyzer(df)
    importance_df = analyzer.calculate_feature_importance(X, y)
    
    st.subheader("🔝 Top Features Influencing Diagnosis")
    n_features = st.slider("Number of features to display:", 5, 20, 10)
    
    top_features = analyzer.get_top_features(n_features)
    
    # Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_features['feature'], top_features['importance'], color='coral')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {n_features} Most Important Features')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with col1:
        st.dataframe(top_features, use_container_width=True, hide_index=True)


def prediction_tab(df, model, scaler, feature_names):
    """Prediction Tab"""
    st.header("🔮 Make Predictions")
    
    if model is None or scaler is None or feature_names is None:
        st.error("❌ Model files not found. Please train the model first.")
        if st.button("Train Model Now"):
            load_and_train_if_needed(df)
            st.rerun()
        return
    
    st.markdown("### Enter patient data for prediction:")
    
    # Get feature values from user
    input_data = {}
    
    # Create columns for input fields
    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        col = cols[idx % 3]
        with col:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            input_data[feature] = st.number_input(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    st.divider()
    
    # Make prediction
    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_data])[feature_names]
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Prediction Result")
            if prediction == 0:
                st.success("✅ **HEALTHY** - No Cancer Detected")
                result_color = "green"
                result_text = "Healthy (No Cancer)"
            else:
                st.error("⚠️ **WARNING** - Cancer Detected")
                result_color = "red"
                result_text = "Cancer Detected"
        
        with col2:
            st.subheader("Input Features Summary")
            st.dataframe(input_df, use_container_width=True)


def main():
    """Main function"""
    header_section()
    
    # Load data and models
    df, model, scaler, feature_names = load_data_and_models()
    
    # Check if models exist, if not train them
    load_and_train_if_needed(df)
    
    # Sidebar navigation
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📍 Navigation",
        [
            "📈 Data Overview",
            "🔬 Exploratory Analysis",
            "🎯 Model Performance",
            "⭐ Feature Importance",
            "🔮 Make Predictions"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 About This App
    
    This application uses machine learning to predict cancer diagnosis based on medical measurements.
    
    **Features:**
    - Data exploration and visualization
    - Model training and comparison
    - Feature importance analysis
    - Real-time predictions
    
    **Model:** Random Forest Classifier
    **Status:** ✅ Ready
    """)
    
    # Route to selected page
    if page == "📈 Data Overview":
        data_overview_tab(df)
    elif page == "🔬 Exploratory Analysis":
        exploratory_analysis_tab(df)
    elif page == "🎯 Model Performance":
        model_performance_tab(df)
    elif page == "⭐ Feature Importance":
        feature_importance_tab(df)
    elif page == "🔮 Make Predictions":
        prediction_tab(df, model, scaler, feature_names)


if __name__ == "__main__":
    main()
