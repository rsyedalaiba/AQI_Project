# **SCRIPT:** YEO–JOHNSON TRANSFORMATION AND TRANSFORMER SAVING
# **AUTHOR:** Syeda Laiba Rehman
# **PURPOSE:**
#   - Load the cleaned AQI dataset
#   - Apply Yeo–Johnson transformation to normalize numeric features
#   - Save the fitted transformer model for all columns used in training (yeo_transformer.pkl) for reuse
#   - Save a new transformed version of the dataset
# **OUTPUTS:**
#   - AQI_Project/models/yeo_transformer.pkl
#   - AQI_Project/data/KARACHI-AQI-RECORDS-2023-2025-TRANSFORMED.csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew
import joblib  #  For saving the transformer

# Load data
cleaned_df = pd.read_csv("AQI_Project/data/KARACHI-AQI-RECORDS-2023-2025-CLEANED.csv")

# Select numeric columns
numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns

# Before transformation
df_before = cleaned_df[numeric_cols].copy()
print("Skewness before:\n", df_before.apply(lambda x: round(skew(x.dropna()), 3)))

# Apply Yeo–Johnson transformation
pt = PowerTransformer(method='yeo-johnson')
df_after = pd.DataFrame(pt.fit_transform(df_before), columns=numeric_cols)

#  Save the fitted transformer model
joblib.dump(pt, "AQI_Project/models/yeo_transformer.pkl")
print("\nYeo–Johnson transformer saved successfully!")

# Replace numeric columns with transformed data
cleaned_df[numeric_cols] = df_after

# Save transformed dataset
transformed_csv = "AQI_Project/data/KARACHI-AQI-RECORDS-2023-2025-TRANSFORMED.csv"
cleaned_df.to_csv(transformed_csv, index=False)
print("\nTransformed dataset saved successfully!")
