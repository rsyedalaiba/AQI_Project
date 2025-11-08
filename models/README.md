**Models Directory**<br>
This folder contains all trained and saved models used in the AQI Prediction Project. <br> 
Each model or transformer file here is generated after training or preprocessing steps.<br>

**Files Overview**<br>

**1. best_lightgbm_multioutput.pkl**<br><br>
**Description:** Trained LightGBM model for multi-output AQI prediction (24h, 48h, 72h ahead).<br>
**Input features:** 15 engineered features from 'ENGINEERED.csv'.<br>
**Output:** Transformed AQI values (before inverse scaling).<br>
**Usage:** Loaded in 'app.py' or prediction scripts for generating AQI forecasts.<br>

**2. yeo_target_only_us_aqi.pkl**<br><br>
**Description:** Fitted Yeo–Johnson transformer on the raw 'us_aqi' values from Cleaned csv.<br>
**Purpose:** Used for inverse-transforming predicted values back to the real AQI scale.<br>
**Usage:** Loaded for converting model predictions from decimal/negative to real AQI integers.<br>

**3. yeo_transformer.pkl**<br><br>
**Description:** Yeo–Johnson transformer fitted on all numeric columns of the cleaned dataset.<br>
**Purpose:** Can be used to normalize numeric data.<br>

**Model Version Info**<br><br>
**Algorithm used:** LightGBM (MultiOutputRegressor)<br>
**Framework:** scikit-learn + joblib<br>
