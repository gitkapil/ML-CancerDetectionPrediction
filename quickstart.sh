#!/bin/bash
# Quick Start Script for Cancer Diagnosis Prediction System

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Cancer Diagnosis Prediction System - Quick Start              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python installation..."
python_version=$(python --version 2>&1)
echo "✓ $python_version"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run pipeline
echo "🚀 Running ML pipeline..."
echo "   Training models and analyzing data..."
python run_pipeline.py
echo ""

# Launch dashboard
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Launching Streamlit Dashboard...                              ║"
echo "║  Open your browser at: http://localhost:8501                   ║"
echo "║  Press Ctrl+C to stop the dashboard                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

streamlit run app.py
