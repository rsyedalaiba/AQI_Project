# AQI_Project
This repository contains a project for predicting Air Quality Index (AQI) using machine learning.<br><br>
## Project Overview

- **Goal**: Build a model to predict AQI levels based on atmospheric and environmental features (e.g., pollutant concentrations, meteorological data).  
- **Key Components**:  
  - Data ingestion & cleaning  
  - Feature engineering & selection  
  - Model training & evaluation  
  - Web application interface for live predictions

## How to Use This Project (Overview)

This project predicts Air Quality Index (AQI) using machine learning models. Here’s a high-level view of how it works:

1. **Data Preparation**  
   - Raw air quality and weather data are processed and cleaned.  
   - Features like rolling averages, lag values, and temporal indicators are generated.

2. **Model Training & Evaluation**  
   - The pipeline trains machine learning models to predict AQI.  
   - Model performance is evaluated using metrics such as R² and RMSE.  
   - SHAP plots are generated to explain predictions.

3. **Making Predictions**  
   - Pre-trained models are saved in the `models/` folder.  
   - Users can input recent air quality and weather data to get AQI forecasts.  
   - Reverse transformations are applied if needed to get predictions in original scale.

4. **Optional Web Interface**  
   - A Flask / Streamlit app can be run locally for a simple dashboard.  
   - Users can view predicted AQI for the next few days and explore trends visually.

> **Note:** The project requires Python 3.12.12 or later. To fetch live data from Hopsworks or other APIs, you must create your own API key. The pipeline will not run fully without it.
