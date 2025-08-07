# Algerian_forest_fires

A self-developed machine learning project to predict forest fire occurrences in two major regions of Algeria — Bejaia and Sidi Bel-abbes — using weather and environmental data from the summer season of 2012.

## 🔥 Project Description

This repository contains a binary classification model that predicts whether a forest fire is likely to occur based on meteorological features like temperature, humidity, wind speed, rainfall, and calculated fire weather indices (FFMC, DMC, DC, ISI, BUI, FWI).

The model was built **from scratch** using hands-on ML workflows including data preprocessing, EDA, model training, evaluation, and interpretation.

---

## 📁 Dataset Overview

- **Instances**: 244 rows (138 fire, 106 no-fire)
- **Features**:
  - Temperature (°C)
  - Relative Humidity (%)
  - Wind Speed (km/h)
  - Rainfall (mm)
  - FFMC, DMC, DC, ISI, BUI, FWI (fire weather indices)
- **Target**: `Classes` → Fire / Not Fire

---


## 📌 Project Workflow

```bash
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering and Correlation Mapping
4. Model Building:
   - Logistic Regression
   - Random Forest
   - Ridge Classifier
5. Evaluation using Accuracy, Precision, Recall, F1-Score
6. Final Model Selection and Summary Insights



# Clone the repository
git clone https://github.com/ItsYourAnirban/Algerian_forest_fires.git
cd Algerian_forest_fires

# Install required packages
pip install -r requirements.txt

# Run the main script
python main.py
