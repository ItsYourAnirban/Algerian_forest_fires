# Algerian_forest_fires

A machine learning project to predict forest fire occurrences in two major regions of Algeria, Bejaia and Sidi Bel-abbes, using meteorological and environmental data collected during the summer of 2012.

## 🔥 Project Description

This project implements a binary classification model to determine the likelihood of a forest fire based on key weather-related features such as temperature, humidity, wind speed, rainfall, and fire weather indices (FFMC, DMC, DC, ISI, BUI, FWI).

The pipeline includes data preprocessing, exploratory data analysis (EDA), model training using multiple algorithms, and evaluation using standard classification metrics.

---

## 📁 Dataset Overview

- **Total Records**: 244 (138 fire, 106 no-fire)
- **Features**:
  - Temperature (°C)
  - Relative Humidity (%)
  - Wind Speed (km/h)
  - Rainfall (mm)
  - FFMC, DMC, DC, ISI, BUI, FWI
- **Target Variable**: `Classes` → Fire / Not Fire

---

## 📌 Project Workflow

```bash
1. Data Preprocessing and Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training:
   - Logistic Regression
   - Random Forest
   - Ridge Classifier
5. Model Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
6. Result Analysis and Summary



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
