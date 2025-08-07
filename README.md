# Algerian Forest Fires Prediction Pipeline

## Overview
This project implements a structured machine learning pipeline to analyze and predict forest fire occurrences in Algeria using meteorological data. The pipeline covers data ingestion, cleaning, exploratory data analysis (EDA), modeling, evaluation, and deployment-ready components.

## Dataset
- **Source**: Algerian Forest Fires dataset containing weather metrics from Bejaia and Sidi‑Bel Abbes regions (June–September 2012), comprising ~244 samples with features like temperature, humidity, wind speed, rainfall, and fire occurrence labels.
- **Files**:
  - `Algerian_forest_fires_dataset_UPDATE.csv` – raw combined dataset
  - `Algerian_forest_fires_cleaned_dataset.csv` – processed and cleaned version

## Key Components
- **Model Training Notebook** (`Model Training.ipynb`): End-to-end pipeline including loading data, preprocessing, model training, and evaluation.
- Other notebooks demonstrating regression techniques:
  - `Practical Simple Linear Regression.ipynb`
  - `Polynomial Regression Implementation.ipynb`
  - `Multiple Linear Regression‑ Economics Dataset.ipynb`
  - `Ridge, Lasso Regression.ipynb`
- Supporting documents:
  - `Ridge,Lasso And Elasticnet.pdf`
  - `Types Of Cross Validation.pdf`

## Workflow
1. **Data Loading**: Read the raw `.csv` containing weather and fire indicators.
2. **Cleaning & Preprocessing**: Handle missing values, standardize formatting, and enhance feature readability.
3. **Exploratory Analysis**: Generate visualizations and compute correlations to understand feature relationships.
4. **Modeling**:
   - Train and compare models such as Logistic Regression, Random Forest, and others.
   - Tune hyperparameters and select the best performing model.
5. **Evaluation**: Use metrics like accuracy, precision, recall, F1‑score, and ROC‑AUC.
6. **Future Steps**: Consider model deployment or integration into real-time monitoring tools.

## Getting Started

```bash
# Clone repository
git clone <repo_url>
cd <repository_folder>

# (Optional) Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and explore notebooks
jupyter notebook
