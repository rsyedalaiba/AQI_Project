# Models Directory
This folder contains all trained and saved models used in the AQI Prediction Project.  
Each model or transformer file here is generated after training or preprocessing steps.

## Files Overview

### `best_lightgbm_multioutput.pkl`
- **Description:** Trained LightGBM model for multi-output AQI prediction (24h, 48h, 72h ahead).
- **Input features:** 15 engineered features from `ENGINEERED.csv`.
- **Output:** Transformed AQI values (before inverse scaling).
- **Usage:** Loaded in `app.py` or prediction scripts for generating AQI forecasts.

### `yeo_target_only_us_aqi.pkl`
- **Description:** Fitted Yeo–Johnson transformer on the raw `us_aqi` values from `Cleaned.csv`.
- **Purpose:** Used for inverse-transforming predicted values back to the real AQI scale.
- **Usage:** Loaded in `aqi_prediction_inverse.py` for converting model predictions from decimal/negative to real AQI integers.

### `yeo_transformer.pkl`
- **Description:** Yeo–Johnson transformer fitted on all numeric columns of the cleaned dataset.
- **Purpose:** Can be used to normalize numeric data.

## Model Version Info
- **Algorithm used:** LightGBM (MultiOutputRegressor)
- **Framework:** scikit-learn + joblib
