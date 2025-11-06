# -*- coding: utf-8 -*-
"""Hopsworks.ipynb
------------------
This notebook connects to the Hopsworks Feature Store,
creates feature groups (offline & online), and uploads AQI data.

Sections:
- Import Libraries
- Connect to Hopsworks
- Create Feature Groups
- Upload Data
- Read & Verify Records
"""

**IMPORT LIBRARIES**
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install hopsworks

# Commented out IPython magic to ensure Python compatibility.
# %pip install confluent-kafka

# Commented out IPython magic to ensure Python compatibility.
# %pip install confluent-kafka

"""**CONNECT TO HOPSWORKS**"""

import hopsworks

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

print("Connected to Hopsworks project:", project.name)

import pandas as pd

data_path = "KARACHI-AQI-RECORDS-2023-2025-ENGINEERED.csv"
df = pd.read_csv(data_path)
df.tail()

"""**CREATE OFFLINE FEATURE GROUP**"""

import pandas as pd

data_path = "KARACHI-AQI-RECORDS-2023-2025-ENGINEERED.csv"
df = pd.read_csv(data_path)

df['time'] = pd.to_datetime(df['time'])

# Drop the target column
features_df = df.drop(columns=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])

aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features_engineered",
    version=1,
    description="Engineered Karachi AQI features (after skewness correction)",
    primary_key=["time"],
    event_time="time",
    online_enabled=False
)

# Confirm it's created
print("Feature group created:", aqi_fg)

"""**RUN TO SAFELY ACCESS FEATURE GROUP WITHOUT RE-CREATION**"""

data_path = "KARACHI-AQI-RECORDS-2023-2025-ENGINEERED.csv"
df = pd.read_csv(data_path)
df['time'] = pd.to_datetime(df['time'])

features_df = df.drop(columns=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'], errors='ignore')
fg_name = "aqi_features_engineered"
fg_version = 1

try:
    aqi_fg = fs.get_feature_group(name=fg_name, version=fg_version)
    print(f"Existing feature group found: {fg_name}_v{fg_version}")
except:
    aqi_fg = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="Engineered Karachi AQI features (after skewness correction)",
        primary_key=["time"],
        event_time="time",
        online_enabled=False
        )
print("Feature group ready:", aqi_fg)

"""**UPLOAD CSV DATA TO HOPSWORKS**"""

df['time'] = pd.to_datetime(df['time'])

features_df = df.drop(columns=['target_aqi_24h', 'target_aqi_48h', 'target_aqi_72h'])

aqi_fg.insert(features_df)
print(" CSV uploaded successfully to Hopsworks Feature Store!")

"""**ONLINE FEATURE GROUP**"""

import os

project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
folders = ['AQI_Project/data', 'AQI_Project/models', 'AQI_Project/notebooks']

for folder in folders:
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)
    #open(os.path.join(folder, '.gitkeep'), 'a').close()

DATA_PATH = os.path.join(project_root, 'AQI_Project', 'data', 'KARACHI-AQI-RECORDS-2023-TO-2025.csv')

print("Folder structure created successfully!\n")
print("\nFolder Tree:")
for folder in folders:
    print(" -", folder)

import pandas as pd

data_path = "AQI_Project/data/ENGINEERED.csv"
df = pd.read_csv(data_path)
df.tail()

"""**CREATE FEATURE GROUP**"""

fg_name = "aqi_features_engineered"
fg_version = 2

aqi_fg_online_v2 = fs.create_feature_group(
    name=fg_name,
    version=fg_version,
    description="Online-enabled version with top 15 features",
    primary_key=["time"],
    event_time="time",
    online_enabled=True
)
print("Feature group ready:", aqi_fg_online_v2)
print(f"Created new online-only feature group: {aqi_fg_online_v2}")

"""**RUN TO SAFELY ACCESS FEATURE GROUP WITHOUT RE-CREATION**"""

fg_name = "aqi_features_engineered"
fg_version = 2

try:
    aqi_fg_online_v2 = fs.get_feature_group(name=fg_name, version=fg_version, online_enabled=True)
    print(f"Existing feature group found: {fg_name}_v{fg_version}")
except:
    aqi_fg_online_v2 = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="Online-enabled version with top 15 features",
        primary_key=["time"],
        event_time="time",
        online_enabled=True
        )
print("Feature group ready:", aqi_fg_online_v2)

"""**FOR UPLOADING DATA FIRST TIME**"""

data_path = "AQI_Project/data/ENGINEERED.csv"
df = pd.read_csv(data_path)
df['time'] = pd.to_datetime(df['time'])

# Top 15 features
online_features = [
        'pm10', 'pm2_5','us_aqi', 'day', 'month', 'year',
       'day_of_week', 'temp_roll24', 'humidity_roll24', 'wind_roll24',
       'AQI_roll24', 'AQI_roll_std24', 'AQI_lag24', 'AQI_roll3',
       'AQI_trend_24h'
    ]

online_df = df[['time'] + online_features]
online_df['time'] = online_df['time'].astype('int64') // 10**9

aqi_fg_online_v2.insert(online_df, write_options={"wait_for_job": True})
print(" CSV uploaded successfully to Hopsworks Feature Store!")

"""**UPLOAD DATA WITHOUT DUPLICATES**"""

import hopsworks
import pandas as pd

# --- Connect to Hopsworks ---
project = hopsworks.login()
fs = project.get_feature_store()

# --- Load new CSV data ---
data_path = "AQI_Project/data/ENGINEERED.csv"
df = pd.read_csv(data_path)
df['time'] = pd.to_datetime(df['time'])

# --- Select features ---
online_features = [
    'pm10', 'pm2_5','us_aqi', 'day', 'month', 'year',
    'day_of_week', 'temp_roll24', 'humidity_roll24', 'wind_roll24',
    'AQI_roll24', 'AQI_roll_std24', 'AQI_lag24', 'AQI_roll3',
    'AQI_trend_24h'
]

# --- Prepare for upload ---
online_df = df[['time'] + online_features]
online_df['time'] = online_df['time'].astype('int64') // 10**9  # UNIX timestamp

# --- Access feature group ---
aqi_fg_online_v2 = fs.get_feature_group("aqi_features_engineered", version=2)

try:
    # Try to read the existing feature group (latest snapshot)
    existing = aqi_fg_online_v2.read()
    existing_times = set(existing['time'].astype(int))
    print(f"Existing records found: {len(existing_times)}")
except Exception as e:
    # If FG is empty or read fails, start fresh
    existing_times = set()
    print("No existing records found — creating initial snapshot.")

# --- Remove duplicates before uploading ---
new_df = online_df[~online_df['time'].isin(existing_times)]

# --- Upload only new data ---
if not new_df.empty:
    aqi_fg_online_v2.insert(new_df, write_options={"wait_for_job": True})
    print(f" {len(new_df)} new records uploaded successfully!")
else:
    print(" No new records to upload — all data already exists.")

"""**TO VIEW RECORDS IN FEATURE GROUP**"""

# Read the online feature group
import pandas as pd

fg_name = "aqi_features_engineered"
fg_version = 2

aqi_fg_online_v2 = fs.get_feature_group(name=fg_name, version=fg_version)

df_features = aqi_fg_online_v2.read()

# Sort by 'time'
df_features = df_features.sort_values(by="time")

# Convert 'time' from epoch seconds back to datetime
df_features['time'] = pd.to_datetime(df_features['time'], unit='s')

# Display first few rows
df_features.tail()
