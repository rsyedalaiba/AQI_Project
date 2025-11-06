import os
import requests
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
import hopsworks
from sklearn.preprocessing import PowerTransformer

# CONFIGURATION
LAT, LON = 24.8607, 67.0011  # Karachi, Pakistan
TIMEZONE = "auto"

# --- Load Hopsworks credentials from environment variables ---
PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT")
API_KEY = os.getenv("HOPSWORKS_API_KEY")

# FETCH FUNCTION
def fetch_data():
    """Fetch AQI and weather data for the last 3 days, filtered to full-day 23:00 hours."""
    TODAY = datetime.utcnow().date()
    START_DATE = (TODAY - timedelta(days=3)).strftime('%Y-%m-%d')
    END_DATE = TODAY.strftime('%Y-%m-%d')

    print(f"Fetching data for {START_DATE} → {END_DATE}")

    # --- AQI data ---
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE
    }
    aqi_response = requests.get(aqi_url, params=aqi_params).json()
    aqi_df = pd.DataFrame(aqi_response["hourly"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"]).dt.tz_localize(None)

    # --- Weather data ---
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "timezone": TIMEZONE
    }

    responses = openmeteo.weather_api(weather_url, params=weather_params)
    response = responses[0]
    hourly = response.Hourly()

    weather_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy()
    }
    weather_df = pd.DataFrame(weather_data)
    weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

    # Merge AQI + weather
    combined_df = pd.merge(aqi_df, weather_df, on="time", how="inner")
    combined_df.sort_values(by="time", inplace=True)

    # --- Filter incomplete last day hours (keep data until previous day's 23:00) ---
    latest_time = combined_df["time"].max()
    print(f"Latest available record: {latest_time}")

    # If latest hour < 23:00, drop partial last day
    if latest_time.hour < 23:
        cutoff_date = (latest_time - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)
        combined_df = combined_df[combined_df["time"] <= cutoff_date]
        print(f"Filtered to full-day data ending at {cutoff_date}")
    else:
        print("Data already complete up to 23:00 — no filtering applied.")

    print(f"Final records after filtering: {len(combined_df)}")
    return combined_df


# CLEAN FUNCTION
def clean_data(df):
    df = df.drop_duplicates().sort_values("time")
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    print("Cleaned data.")
    return df


# TRANSFORMATION FUNCTION
def transform_data(df):
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    pt = PowerTransformer(method="yeo-johnson")
    df[numeric_cols] = pt.fit_transform(df[numeric_cols])
    print("Transformed data (normalized).")
    return df


# FEATURE ENGINEERING FUNCTION
def engineer_features(df):
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["day_of_week"] = df["time"].dt.dayofweek

    df["temp_roll24"] = df["temperature_2m"].rolling(window=24, min_periods=1).mean()
    df["humidity_roll24"] = df["relative_humidity_2m"].rolling(window=24, min_periods=1).mean()
    df["wind_roll24"] = df["wind_speed_10m"].rolling(window=24, min_periods=1).mean()
    df["AQI_roll24"] = df["us_aqi"].rolling(window=24, min_periods=1).mean()
    df["AQI_roll_std24"] = df["us_aqi"].rolling(window=24, min_periods=1).std()
    df["AQI_lag24"] = df["us_aqi"].shift(24)
    df["AQI_trend_24h"] = df["us_aqi"] - df["us_aqi"].shift(24)
    df["AQI_roll3"] = df["us_aqi"].rolling(window=3, min_periods=1).mean()

    df.dropna(inplace=True)
    print("Engineered 15 key features.")
    return df


def upload_to_hopsworks_online(df: pd.DataFrame):
    print("Uploading to Hopsworks...")

    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    aqi_fg_online_v2 = fs.get_feature_group("aqi_features_engineered", version=2)

    # --- Convert time to Unix ---
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].astype('int64') // 10**9

    # --- Ensure correct dtypes for Hopsworks schema ---
    int_columns = ['day', 'month', 'year', 'day_of_week']
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].astype('int64')

    # --- Optional: print dtype check for debugging ---
    print("\n Dtypes before upload:")
    print(df[int_columns + ['time']].dtypes)
    print("-" * 60)

    online_features = [
        'pm10', 'pm2_5', 'us_aqi', 'day', 'month', 'year',
        'day_of_week', 'temp_roll24', 'humidity_roll24', 'wind_roll24',
        'AQI_roll24', 'AQI_roll_std24', 'AQI_lag24', 'AQI_roll3',
        'AQI_trend_24h'
    ]
    upload_df = df[['time'] + online_features]

    try:
        existing = aqi_fg_online_v2.read()
        existing_times = set(existing['time'].astype(int))
        print(f"Existing records: {len(existing_times)}")
    except Exception:
        existing_times = set()
        print("No existing records found — creating new dataset.")

    new_df = upload_df[~upload_df['time'].isin(existing_times)]

    if not new_df.empty:
        aqi_fg_online_v2.insert(new_df, write_options={"wait_for_job": True})
        print(f"{len(new_df)} new records uploaded successfully!")
    else:
        print("No new records to upload — all data already exists.")

# MAIN
def main():
    df = fetch_data()
    df = clean_data(df)
    df = transform_data(df)
    df = engineer_features(df)
    upload_to_hopsworks_online(df)
    print("Daily AQI pipeline completed successfully!")


# ENTRY POINT
if __name__ == "__main__":
    main()
