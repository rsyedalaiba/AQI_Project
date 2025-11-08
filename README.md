# AQI_Project

## Overview
This project predicts hourly AQI (Air Quality Index) values for the next 72 hours using multiple machine learning models and an automated pipeline.

Key components of the system include:  

- Machine learning models (Random Forest, XGBoost, LightGBM) for AQI prediction  
- Hopsworks Feature Store for historical and real-time data management  
- GitHub Actions for CI/CD automation  
- Streamlit frontend for visualization and interactive dashboards  

## System Architecture

### Data Source & Feature Store (Hopsworks)
- Historical AQI and weather features (PM2.5, PM10, temperature, humidity, rainfall, wind speed, etc.) are stored in Hopsworks.  
- Automated scripts fetch the latest features before each prediction.  
- Version-controlled datasets ensure reproducibility and consistency.

### Model Training & Forecasting
- Ensemble ML algorithms used:
  - **Random Forest (RF)**
  - **XGBoost (XGB)**
  - **LightGBM (LGBM)**
- Models predict hourly AQI for the next 3 days.  

### CI/CD Automation (GitHub Actions)  
**Pipeline steps:**
- Fetch new data from APIs  
- Clean, Transform and perform Feature Engineering  
- Save new data to hopsworks. This new data is then used for predictions

### Streamlit Frontend
- Displays 3-day hourly AQI predictions with:  
- Color-coded categories for AQI levels  
- Trend graphs using Plotly  
- Daily health recommendations  

### Observations
- Models perform best for short-term forecasts.  
- LightGBM performed the best with 96% R², providing high accuracy and generalization.  
- XGBoost and Random Forest gives stable baseline predictions.  
- Ensemble outputs deliver smoother and more reliable 3-day trends.  


## Tech Stack

**Frontend:** Streamlit, Plotly 
**Backend:** Python, scikit-learn 
**ML Models:** RandomForest, XGBoost, LightGBM 
**Feature Store:** Hopsworks 
**CI/CD:** GitHub Actions 
**Explainability:** SHAP 
**Visualization:** Plotly, Matplotlib 
**Utilities:** Pandas, NumPy, python-dotenv 

## Workflow Summary
1. **Data Fetching:** Retrieve features from Hopsworks . 
2. **Model Training:** Train RF, XGB, LGBM using AQI + weather features.  
3. **Model Evaluation:** Compute R², MAE, RMSE for each model.
4. **Model Selection:** Selects Best Model.
5. **Automation:** Add new data in Hopsworks daily for predcition.
6. **Dashboard:** Interactive 3-day AQI forecasts rendered using Streamlit and Plotly; ngrok is used to expose the local dashboard for remote access.  

## Author
**Syeda Laiba Rehman**  
Bahria University, CS Department  
Project: AQI Prediction & Forecasting

> **Note:** To fetch live data from Hopsworks or other APIs, you must create your own API key. The pipeline will not run fully without it.
