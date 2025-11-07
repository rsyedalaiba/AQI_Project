# -*- coding: utf-8 -*-
"""APP.ipynb

**Author:** Syeda Laiba Rehman
**Project:** Air Quality Index (AQI) Prediction System

**🌫️ Live AQI Dashboard**
Streamlit web application for real-time Air Quality Index predictions

**Description:**
Real-time dashboard that predicts AQI levels for the next 72 hours using 
machine learning model(LightGBM) trained on historical air quality and weather data.

**Features:**
- 24H, 48H, and 72H AQI predictions with color-coded visualization
- Day/Night indicators for each hour
- Interactive trend analysis charts
- Feature importance analysis using SHAP values
- Real-time statistics with trend indicators
- Daily Health Summary

**Technologies:**
- Streamlit (Frontend)
- Hopsworks (Feature Store)
- LightGBM (Machine Learning)
- SHAP (Model Explainability)
- Plotly (Visualizations)

**Usage:**
1. Install dependencies: pip install -r requirements.txt
2. Run the app: streamlit run dashboard.py
3. Access via: http://localhost:8501

**Environment Variables:**
- HOPSWORKS_API_KEY: Your Hopsworks API key for feature store access

**IMPORT LIBRARIES**
"""

# STEP 1
!pip install hopsworks flask streamlit pyngrok joblib lightgbm

"""**CREATE FOLDERS**"""

import os

project_root = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
folders = ['AQI_Project/data', 'AQI_Project/models', 'AQI_Project/notebooks']

for folder in folders:
    os.makedirs(os.path.join(project_root, folder), exist_ok=True)

DATA_PATH = os.path.join(project_root, 'AQI_Project', 'data', 'KARACHI_AQI_WEATHER_2023_TO_2025.csv')

print("Folder structure created successfully!\n")
print("Project Root:", project_root)
print("Data Path:", DATA_PATH)
print("\nFolder Tree:")
for folder in folders:
    print(" -", folder)

!python flask_api.py &

# STEP 2
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHENTICATION_TOKEN_HERE")

# STEP 3
flask_tunnel = ngrok.connect(5000)
print("Flask API URL: ",flask_tunnel.public_url)

"""**CREATE STREAMLIT APP**"""

# Commented out IPython magic to ensure Python compatibility.
# # STEP 4 - UPDATED VERSION
# %%writefile dashboard.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import hopsworks
# from datetime import datetime, timedelta
# import os
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# 
# st.set_page_config(page_title="Live AQI Dashboard", layout="wide")
# 
# st.title("🌫️ Live AQI Dashboard")
# 
# # ---------------------- DAILY HEALTH SUMMARY FUNCTION ----------------------
# 
# def display_daily_health_summary(preds_24):
#     """Display daily health summary with best and worst times for activities"""
#     st.markdown("---")
#     st.subheader("📅 Daily Health Summary")
# 
#     # Consider only reasonable hours (6 AM to 10 PM) - when people are typically active
#     reasonable_hours = list(range(6, 23))  # 6:00 to 22:00
# 
#     if len(preds_24) >= 23:
#         reasonable_aqis = [preds_24[h] for h in reasonable_hours]
#         reasonable_hourly_aqis = [(h, preds_24[h]) for h in reasonable_hours]
# 
#         # Find best and worst times (only from reasonable hours)
#         if reasonable_hourly_aqis:
#             best_hour, best_aqi = min(reasonable_hourly_aqis, key=lambda x: x[1])
#             worst_hour, worst_aqi = max(reasonable_hourly_aqis, key=lambda x: x[1])
# 
#             col1, col2 = st.columns(2)
# 
#             with col1:
#                 st.markdown(
#                     f"""
#                     <div style="background-color: #00e40020; padding: 20px; border-radius: 10px; border-left: 5px solid #00e400; margin-bottom: 20px;">
#                         <div style="font-weight: bold; font-size: 18px; color: #00e400; margin-bottom: 10px;">🌿 Best Time for Outdoor Activities</div>
#                         <div style="font-size: 24px; font-weight: bold; margin: 15px 0; color: #333;">{best_hour:02d}:00 - {best_hour+1:02d}:00</div>
#                         <div style="font-size: 16px; color: #666; background: white; padding: 8px; border-radius: 5px; margin: 10px 0;">
#                             <strong>AQI: {best_aqi:.0f}</strong> ({get_aqi_category(best_aqi)})
#                         </div>
#                         <div style="font-size: 14px; color: #555; margin-top: 12px;">
#                         🚴 Perfect for walking, exercise, and outdoor work
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
# 
#             with col2:
#                 st.markdown(
#                     f"""
#                     <div style="background-color: #ff7e0020; padding: 20px; border-radius: 10px; border-left: 5px solid #ff7e00; margin-bottom: 20px;">
#                         <div style="font-weight: bold; font-size: 18px; color: #ff7e00; margin-bottom: 10px;">⚠️ Avoid Outdoor Activities</div>
#                         <div style="font-size: 24px; font-weight: bold; margin: 15px 0; color: #333;">{worst_hour:02d}:00 - {worst_hour+1:02d}:00</div>
#                         <div style="font-size: 16px; color: #666; background: white; padding: 8px; border-radius: 5px; margin: 10px 0;">
#                             <strong>AQI: {worst_aqi:.0f}</strong> ({get_aqi_category(worst_aqi)})
#                         </div>
#                         <div style="font-size: 14px; color: #555; margin-top: 12px;">
#                         🚨 Limit time outside during this period
#                         </div>
#                     </div>
#                     """,
#                     unsafe_allow_html=True
#                 )
# 
#         # Daily recommendations based on overall air quality
#         avg_reasonable_aqi = np.mean(reasonable_aqis)
# 
#         st.markdown("#### 💡 Daily Recommendations")
# 
#         if avg_reasonable_aqi <= 50:
#             st.markdown(
#                 """
#                 <div style="background-color: #00e40020; padding: 15px; border-radius: 8px; border-left: 4px solid #00e400;">
#                     <div style="font-weight: bold; color: #00e400;">✅ Excellent Day Ahead!</div>
#                     <div style="font-size: 14px; color: #555; margin-top: 8px;">
#                         Perfect for all outdoor activities! Great day for exercise, sports, and outdoor work.
#                         Windows can be open for fresh air ventilation.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         elif avg_reasonable_aqi <= 100:
#             st.markdown(
#                 """
#                 <div style="background-color: #ffff0020; padding: 15px; border-radius: 8px; border-left: 4px solid #ffff00;">
#                     <div style="font-weight: bold; color: #b3a600;">🌤️ Good Day for Activities</div>
#                     <div style="font-size: 14px; color: #555; margin-top: 8px;">
#                         Generally good air quality throughout the day. Most people can enjoy outdoor activities.
#                         Sensitive individuals should monitor symptoms during prolonged exertion.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         elif avg_reasonable_aqi <= 150:
#             st.markdown(
#                 """
#                 <div style="background-color: #ff7e0020; padding: 15px; border-radius: 8px; border-left: 4px solid #ff7e00;">
#                     <div style="font-weight: bold; color: #ff7e00;">🟡 Moderate Conditions</div>
#                     <div style="font-size: 14px; color: #555; margin-top: 8px;">
#                         Sensitive groups should reduce prolonged outdoor activities.
#                         People with asthma, children, and elderly should take precautions.
#                         Healthy individuals are generally fine for normal activities.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         else:
#             st.markdown(
#                 """
#                 <div style="background-color: #ff000020; padding: 15px; border-radius: 8px; border-left: 4px solid #ff0000;">
#                     <div style="font-weight: bold; color: #ff0000;">🚨 Poor Air Quality Day</div>
#                     <div style="font-size: 14px; color: #555; margin-top: 8px;">
#                         Limit outdoor activities, especially during peak hours.
#                         Sensitive groups should stay indoors.
#                         Consider rescheduling strenuous outdoor activities.
#                     </div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
# 
# # ---------------------- ORIGINAL FUNCTIONS ----------------------
# 
# # Function to create AQI cards with day/night icons below
# def create_aqi_cards(predictions, title, start_hour=0):
#     st.subheader(title)
# 
#     # Create time headers with proper 00:00 format
#     cols = st.columns(24)
#     for i, col in enumerate(cols):
#         hour_display = f"{(start_hour + i) % 24:02d}:00"
#         col.markdown(f"<div style='text-align: center; font-weight: bold; margin-bottom: 5px; font-size: 12px; white-space: nowrap;'>{hour_display}</div>", unsafe_allow_html=True)
# 
#     # Create square cards with AQI values (original layout)
#     cols = st.columns(24)
#     for i, col in enumerate(cols):
#         if i < len(predictions):
#             aqi_value = int(np.round(predictions[i], 0))
# 
#             # Color coding based on AQI value
#             if aqi_value <= 50:
#                 color = "#00e400"  # Good - Green
#             elif aqi_value <= 100:
#                 color = "#ffff00"  # Moderate - Yellow
#             elif aqi_value <= 150:
#                 color = "#ff7e00"  # Unhealthy for Sensitive - Orange
#             elif aqi_value <= 200:
#                 color = "#ff0000"  # Unhealthy - Red
#             elif aqi_value <= 300:
#                 color = "#8f3f97"  # Very Unhealthy - Purple
#             else:
#                 color = "#7e0023"  # Hazardous - Maroon
# 
#             col.markdown(
#                 f"""
#                 <div style="
#                     background-color: {color};
#                     padding: 20px;
#                     border-radius: 5px;
#                     text-align: center;
#                     color: {'white' if aqi_value > 100 else 'black'};
#                     font-weight: bold;
#                     font-size: 16px;
#                     margin: 2px;
#                     height: 60px;
#                     display: flex;
#                     align-items: center;
#                     justify-content: center;
#                 ">
#                     {aqi_value}
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
# 
#     # Create day/night icons row below the AQI cards
#     cols_icons = st.columns(24)
#     for i, col in enumerate(cols_icons):
#         if i < len(predictions):
#             current_hour = (start_hour + i) % 24
# 
#             # Determine day/night icon (7 AM to 5 PM is day, rest is night)
#             if 7 <= current_hour <= 17:  # 7 AM to 5 PM
#                 icon = "☀️"  # Sun icon for day
#                 icon_text = "Day"
#             else:
#                 icon = "🌙"  # Moon icon for night
#                 icon_text = "Night"
# 
#             col.markdown(
#                 f"""
#                 <div style="
#                     text-align: center;
#                     margin-top: 5px;
#                     font-size: 14px;
#                 ">
#                     <div>{icon}</div>
#                     <div style="font-size: 10px; margin-top: 2px;">{icon_text}</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
# 
# # Function to create AQI trend charts
# def create_aqi_trend_charts(preds_24, preds_48, preds_72, dates):
#     st.markdown("---")
#     st.subheader("📈 AQI Trends Analysis")
# 
#     # Create three columns for the charts
#     col1, col2, col3 = st.columns(3)
# 
#     # Prepare data for charts
#     hours = [f"{i:02d}:00" for i in range(24)]
# 
#     with col1:
#         # 24H Trend Chart
#         fig1 = go.Figure()
#         fig1.add_trace(go.Scatter(x=hours, y=preds_24, mode='lines+markers',
#                                 line=dict(color='#1f77b4', width=3),
#                                 marker=dict(size=6),
#                                 name='AQI'))
# 
#         # Add color bands for AQI levels
#         fig1.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
#         fig1.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
#         fig1.add_hrect(y0=101, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
#         fig1.add_hrect(y0=151, y1=200, fillcolor="red", opacity=0.1, line_width=0)
#         fig1.add_hrect(y0=201, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
#         fig1.add_hrect(y0=301, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
# 
#         fig1.update_layout(
#             title=f"24H AQI Trend - {dates[0]}",
#             xaxis_title="Time",
#             yaxis_title="AQI",
#             height=300,
#             showlegend=False,
#             margin=dict(l=40, r=40, t=60, b=40)
#         )
#         st.plotly_chart(fig1, use_container_width=True)
# 
#     with col2:
#         # 48H Trend Chart
#         fig2 = go.Figure()
#         fig2.add_trace(go.Scatter(x=hours, y=preds_48, mode='lines+markers',
#                                 line=dict(color='#ff7f0e', width=3),
#                                 marker=dict(size=6),
#                                 name='AQI'))
# 
#         # Add color bands
#         fig2.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
#         fig2.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
#         fig2.add_hrect(y0=101, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
#         fig2.add_hrect(y0=151, y1=200, fillcolor="red", opacity=0.1, line_width=0)
#         fig2.add_hrect(y0=201, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
#         fig2.add_hrect(y0=301, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
# 
#         fig2.update_layout(
#             title=f"48H AQI Trend - {dates[1]}",
#             xaxis_title="Time",
#             yaxis_title="AQI",
#             height=300,
#             showlegend=False,
#             margin=dict(l=40, r=40, t=60, b=40)
#         )
#         st.plotly_chart(fig2, use_container_width=True)
# 
#     with col3:
#         # 72H Trend Chart
#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(x=hours, y=preds_72, mode='lines+markers',
#                                 line=dict(color='#2ca02c', width=3),
#                                 marker=dict(size=6),
#                                 name='AQI'))
# 
#         # Add color bands
#         fig3.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
#         fig3.add_hrect(y0=51, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
#         fig3.add_hrect(y0=101, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
#         fig3.add_hrect(y0=151, y1=200, fillcolor="red", opacity=0.1, line_width=0)
#         fig3.add_hrect(y0=201, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
#         fig3.add_hrect(y0=301, y1=500, fillcolor="maroon", opacity=0.1, line_width=0)
# 
#         fig3.update_layout(
#             title=f"72H AQI Trend - {dates[2]}",
#             xaxis_title="Time",
#             yaxis_title="AQI",
#             height=300,
#             showlegend=False,
#             margin=dict(l=40, r=40, t=60, b=40)
#         )
#         st.plotly_chart(fig3, use_container_width=True)
# 
# # Function to create feature importance with SHAP (Prediction-specific)
# def create_prediction_importance(model, X_recent, selected_features, date):
#     st.markdown("---")
#     st.subheader(f"🔍 Feature Impact on Today's Prediction ({date})")
# 
#     try:
#         import shap
# 
#         # For MultiOutputRegressor, we need to extract the underlying estimators
#         if hasattr(model, 'estimators_'):
#             # MultiOutputRegressor - use the first estimator (for 24H prediction) or average all
#             underlying_model = model.estimators_[0]  # Use first estimator for 24H prediction
# 
# 
#         else:
#             # Single model case
#             underlying_model = model
# 
#         # Create explainer with the underlying model
#         explainer = shap.TreeExplainer(underlying_model)
# 
#         # Calculate SHAP values for the latest data point
#         shap_values = explainer.shap_values(X_recent.iloc[-1:])
# 
#         # Handle different SHAP value formats
#         if isinstance(shap_values, list):
#             # For multi-class, take the first one (24H prediction)
#             shap_values = shap_values[0]
# 
#         # Ensure we have a 1D array
#         if len(shap_values.shape) > 1:
#             shap_values = shap_values[0]
# 
#         # Create feature names mapping
#         feature_names_map = {
#             'pm10': 'PM10 Levels',
#             'pm2_5': 'PM2.5 Levels',
#             'us_aqi': 'Current AQI',
#             'day': 'Day of Month',
#             'month': 'Month',
#             'year': 'Year',
#             'day_of_week': 'Day of Week',
#             'temp_roll24': '24H Avg Temp',
#             'humidity_roll24': '24H Avg Humidity',
#             'wind_roll24': '24H Avg Wind',
#             'aqi_roll24': '24H Avg AQI',
#             'aqi_roll_std24': '24H AQI Variability',
#             'aqi_lag24': 'Prev Day AQI',
#             'aqi_roll3': '3H Avg AQI',
#             'aqi_trend_24h': '24H AQI Trend'
#         }
# 
#         readable_features = [feature_names_map.get(feature, feature) for feature in selected_features]
# 
#         # Create DataFrame
#         importance_df = pd.DataFrame({
#             'feature': readable_features,
#             'impact': np.abs(shap_values)  # Use absolute values for importance
#         }).sort_values('impact', ascending=True)
# 
#         # Create chart
#         fig = px.bar(
#             importance_df,
#             y='feature',
#             x='impact',
#             orientation='h',
#             title=f"Features Driving Today's AQI Prediction (24H)",
#             height=400,
#             color='impact',
#             color_continuous_scale='reds'
#         )
# 
#         fig.update_layout(
#             xaxis_title="Impact on Prediction (Absolute SHAP Value)",
#             yaxis_title="Features",
#             showlegend=False,
#             margin=dict(l=20, r=20, t=60, b=20)
#         )
# 
#         st.plotly_chart(fig, use_container_width=True)
# 
#         # Also show overall feature importance as fallback
#         if hasattr(underlying_model, 'feature_importances_'):
#             st.markdown("#### 📊 Overall Model Feature Importance (24H)")
# 
#             overall_importance_df = pd.DataFrame({
#                 'feature': readable_features,
#                 'importance': underlying_model.feature_importances_
#             }).sort_values('importance', ascending=True)
# 
#             fig2 = px.bar(
#                 overall_importance_df,
#                 y='feature',
#                 x='importance',
#                 orientation='h',
#                 title="Overall Feature Importance (From Training)",
#                 height=400,
#                 color='importance',
#                 color_continuous_scale='blues'
#             )
# 
#             fig2.update_layout(
#                 xaxis_title="Feature Importance Score",
#                 yaxis_title="Features",
#                 showlegend=False,
#                 margin=dict(l=20, r=20, t=60, b=20)
#             )
# 
#             st.plotly_chart(fig2, use_container_width=True)
# 
#     except ImportError:
#         st.warning("SHAP not installed. Install with: `pip install shap`")
#     except Exception as e:
#         st.warning(f"Could not calculate prediction-specific importance: {e}")
#         # Fallback to overall feature importance
#         show_overall_importance_fallback(model, selected_features)
# 
# # Fallback function to show overall feature importance
# def show_overall_importance_fallback(model, selected_features):
#     st.info("Showing overall feature importance instead of prediction-specific importance")
# 
#     try:
#         # Get the underlying model from MultiOutputRegressor
#         if hasattr(model, 'estimators_'):
#             underlying_model = model.estimators_[0]  # Use first estimator
#         else:
#             underlying_model = model
# 
#         if hasattr(underlying_model, 'feature_importances_'):
#             # Create feature names mapping
#             feature_names_map = {
#                 'pm10': 'PM10 Levels',
#                 'pm2_5': 'PM2.5 Levels',
#                 'us_aqi': 'Current AQI',
#                 'day': 'Day of Month',
#                 'month': 'Month',
#                 'year': 'Year',
#                 'day_of_week': 'Day of Week',
#                 'temp_roll24': '24H Avg Temp',
#                 'humidity_roll24': '24H Avg Humidity',
#                 'wind_roll24': '24H Avg Wind',
#                 'aqi_roll24': '24H Avg AQI',
#                 'aqi_roll_std24': '24H AQI Variability',
#                 'aqi_lag24': 'Prev Day AQI',
#                 'aqi_roll3': '3H Avg AQI',
#                 'aqi_trend_24h': '24H AQI Trend'
#             }
# 
#             readable_features = [feature_names_map.get(feature, feature) for feature in selected_features]
# 
#             importance_df = pd.DataFrame({
#                 'feature': readable_features,
#                 'importance': underlying_model.feature_importances_
#             }).sort_values('importance', ascending=True)
# 
#             fig = px.bar(
#                 importance_df,
#                 y='feature',
#                 x='importance',
#                 orientation='h',
#                 title="Overall Feature Importance (24H Prediction Model)",
#                 height=400,
#                 color='importance',
#                 color_continuous_scale='viridis'
#             )
# 
#             fig.update_layout(
#                 xaxis_title="Feature Importance Score",
#                 yaxis_title="Features",
#                 showlegend=False
#             )
# 
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.error("Could not extract feature importance from the model")
# 
#     except Exception as e:
#         st.error(f"Error in fallback feature importance: {e}")
# 
# # Function to create AQI statistics cards with trend deltas
# def create_statistics_cards(preds_24, preds_48, preds_72, dates):
#     st.markdown("---")
#     st.subheader("📊 AQI Statistics Summary")
# 
#     # Calculate statistics for EACH day
#     stats_24h = {
#         'max': np.max(preds_24),
#         'min': np.min(preds_24),
#         'avg': np.mean(preds_24),
#         'date': dates[0]
#     }
# 
#     stats_48h = {
#         'max': np.max(preds_48),
#         'min': np.min(preds_48),
#         'avg': np.mean(preds_48),
#         'date': dates[1]
#     }
# 
#     stats_72h = {
#         'max': np.max(preds_72),
#         'min': np.min(preds_72),
#         'avg': np.mean(preds_72),
#         'date': dates[2]
#     }
# 
#     # Calculate trends (deltas between days)
#     avg_trend_48h = stats_48h['avg'] - stats_24h['avg']
#     avg_trend_72h = stats_72h['avg'] - stats_48h['avg']
# 
#     max_trend_48h = stats_48h['max'] - stats_24h['max']
#     max_trend_72h = stats_72h['max'] - stats_48h['max']
# 
#     min_trend_48h = stats_48h['min'] - stats_24h['min']
#     min_trend_72h = stats_72h['min'] - stats_48h['min']
# 
#     # Display in columns - one for each day
#     col1, col2, col3 = st.columns(3)
# 
#     with col1:
#         st.subheader(f"24H - {stats_24h['date']}")
# 
#         st.metric(
#             label="Average AQI",
#             value=f"{stats_24h['avg']:.0f}",
#             delta=get_aqi_category(stats_24h['avg']),
#             delta_color="off"
#         )
# 
#         st.metric(
#             label="Peak AQI",
#             value=f"{stats_24h['max']:.0f}",
#             delta="Peak",
#             delta_color="off"
#         )
# 
#         st.metric(
#             label="Lowest AQI",
#             value=f"{stats_24h['min']:.0f}",
#             delta="Lowest",
#             delta_color="off"
#         )
# 
#     with col2:
#         st.subheader(f"48H - {stats_48h['date']}")
# 
#         st.metric(
#             label="Average AQI",
#             value=f"{stats_48h['avg']:.0f}",
#             delta=f"{avg_trend_48h:+.0f}",
#             delta_color="inverse" if avg_trend_48h > 0 else "normal"
#         )
# 
#         st.metric(
#             label="Peak AQI",
#             value=f"{stats_48h['max']:.0f}",
#             delta=f"{max_trend_48h:+.0f}",
#             delta_color="inverse" if max_trend_48h > 0 else "normal"
#         )
# 
#         st.metric(
#             label="Lowest AQI",
#             value=f"{stats_48h['min']:.0f}",
#             delta=f"{min_trend_48h:+.0f}",
#             delta_color="inverse" if min_trend_48h > 0 else "normal"
#         )
# 
#     with col3:
#         st.subheader(f"72H - {stats_72h['date']}")
# 
#         st.metric(
#             label="Average AQI",
#             value=f"{stats_72h['avg']:.0f}",
#             delta=f"{avg_trend_72h:+.0f}",
#             delta_color="inverse" if avg_trend_72h > 0 else "normal"
#         )
# 
#         st.metric(
#             label="Peak AQI",
#             value=f"{stats_72h['max']:.0f}",
#             delta=f"{max_trend_72h:+.0f}",
#             delta_color="inverse" if max_trend_72h > 0 else "normal"
#         )
# 
#         st.metric(
#             label="Lowest AQI",
#             value=f"{stats_72h['min']:.0f}",
#             delta=f"{min_trend_72h:+.0f}",
#             delta_color="inverse" if min_trend_72h > 0 else "normal"
#         )
# 
# # Helper function to get AQI category for delta text
# def get_aqi_category(aqi_value):
#     if aqi_value <= 50:
#         return "Good"
#     elif aqi_value <= 100:
#         return "Moderate"
#     elif aqi_value <= 150:
#         return "Unhealthy Sensitive"
#     elif aqi_value <= 200:
#         return "Unhealthy"
#     elif aqi_value <= 300:
#         return "Very Unhealthy"
#     else:
#         return "Hazardous"
# 
# # Calculate dates for headings
# current_date = datetime.now()
# date_24h = current_date.strftime("%d/%m/%Y")
# date_48h = (current_date + timedelta(days=1)).strftime("%d/%m/%Y")
# date_72h = (current_date + timedelta(days=2)).strftime("%d/%m/%Y")
# dates = [date_24h, date_48h, date_72h]
# 
# # Initialize variables
# use_demo_data = False
# preds_24, preds_48, preds_72 = None, None, None
# model = None
# X_recent = None
# selected_features = [
#     'pm10', 'pm2_5', 'us_aqi', 'day', 'month', 'year',
#     'day_of_week', 'temp_roll24', 'humidity_roll24', 'wind_roll24',
#     'aqi_roll24', 'aqi_roll_std24', 'aqi_lag24', 'aqi_roll3', 'aqi_trend_24h'
# ]
# 
# try:
#     st.markdown("Fetching latest AQI data from Hopsworks & predicting next 3 days...")
# 
#     # --- Connect to Hopsworks
#     api_key = os.getenv("HOPSWORKS_API_KEY", "YOUR_API_KEY_HERE")
# 
#     project = hopsworks.login(api_key_value=api_key, project="AQI_Weather")
#     fs = project.get_feature_store()
#     feature_group = fs.get_feature_group(name="aqi_features_engineered", version=2)
#     df_features = feature_group.read(read_options={"online": True})
# 
#     df_features['time'] = pd.to_datetime(df_features['time'], unit='s')
#     df_features = df_features.sort_values(by='time', ascending=True)
#     df_recent = df_features.tail(24).reset_index(drop=True)
# 
#     # --- Load models
#     try:
#         pt = joblib.load("AQI_Project/models/yeo_target_only_us_aqi.pkl")
#         model = joblib.load("AQI_Project/models/best_lightgbm_multioutput.pkl")
#     except Exception as model_error:
#         st.warning(f"Model files not found: {model_error}")
#         use_demo_data = True
# 
#     if not use_demo_data:
#         X_recent = df_recent[selected_features].copy()
#         X_recent.columns = selected_features
# 
#         # --- Predict
#         predictions = model.predict(X_recent)
#         try:
#             preds_24 = pt.inverse_transform(predictions[:,0].reshape(-1,1)).flatten()
#             preds_48 = pt.inverse_transform(predictions[:,1].reshape(-1,1)).flatten()
#             preds_72 = pt.inverse_transform(predictions[:,2].reshape(-1,1)).flatten()
#         except:
#             preds_24, preds_48, preds_72 = predictions[:,0], predictions[:,1], predictions[:,2]
# 
# except Exception as e:
#     st.warning(f"Could not connect to Hopsworks: {e}")
#     st.info("Using demo data for display purposes")
#     use_demo_data = True
# 
# # If connection failed or demo data requested, use sample data
# if use_demo_data or preds_24 is None:
#     # Generate realistic demo data
#     np.random.seed(42)
#     base_aqi = 45
#     preds_24 = base_aqi + np.random.normal(0, 10, 24)
#     preds_24 = np.clip(preds_24, 20, 300)
# 
#     preds_48 = base_aqi + np.random.normal(5, 12, 24)
#     preds_48 = np.clip(preds_48, 20, 300)
# 
#     preds_72 = base_aqi + np.random.normal(8, 15, 24)
#     preds_72 = np.clip(preds_72, 20, 300)
# 
# # Display predictions in the card format with dynamic dates
# create_aqi_cards(preds_24, f"{date_24h}")
# create_aqi_cards(preds_48, f"{date_48h}")
# create_aqi_cards(preds_72, f"{date_72h}")
# 
# # Show connection status
# if use_demo_data:
#     st.warning(" Currently showing demo data. Real predictions will appear when Hopsworks connection is established.")
# 
# # Add the new charts section
# create_statistics_cards(preds_24, preds_48, preds_72, dates)
# create_aqi_trend_charts(preds_24, preds_48, preds_72, dates)
# 
# # ---------------------- ADDED: DAILY HEALTH SUMMARY ----------------------
# 
# 
# # Only show prediction-specific importance if we have a real model and data
# if model is not None and X_recent is not None and not use_demo_data:
#     create_prediction_importance(model, X_recent, selected_features, date_24h)
# else:
#     st.markdown("---")
#     st.subheader("🔍 Feature Impact Analysis")
#     st.info("Prediction-specific feature importance will be available when connected to Hopsworks with real data.")
# 
# # Display daily health summary before feature importance
# display_daily_health_summary(preds_24)
# 
# # Simple AQI color legend
# st.markdown("---")
# st.subheader("AQI Color Guide")
# legend_cols = st.columns(6)
# 
# legend_data = [
#     ("#00e400", "0-50: Good"),
#     ("#ffff00", "51-100: Moderate"),
#     ("#ff7e00", "101-150: Unhealthy Sensitive"),
#     ("#ff0000", "151-200: Unhealthy"),
#     ("#8f3f97", "201-300: Very Unhealthy"),
#     ("#7e0023", "301+: Hazardous")
# ]
# 
# for i, (color, text) in enumerate(legend_data):
#     with legend_cols[i]:
#         st.markdown(
#             f"""
#             <div style="background-color: {color}; padding: 8px; border-radius: 5px; text-align: center; font-size: 12px; margin: 2px;">
#                 <strong>{text}</strong>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
# 
# # Add icon legend
# st.markdown("---")
# st.subheader("Icon Guide")
# icon_cols = st.columns(2)
# with icon_cols[0]:
#     st.markdown("**☀️ Day (7:00 AM - 5:00 PM)**")
# with icon_cols[1]:
#     st.markdown("**🌙 Night (6:00 PM - 6:00 AM)**")
# 
# st.markdown("""
# ---
# **Powered by:** Hopsworks • Streamlit • Ngrok
# """)

"""**EXECUTE APP**"""

# STEP 5
from pyngrok import ngrok

public_url = ngrok.connect(8501)
print(f"🌐 Your live AQI dashboard URL: {public_url}")

!streamlit run dashboard.py &

# Kill previous Flask if running
!kill -9 $(lsof -t -i:5000) || echo "No previous Flask process"

# Run Flask server
get_ipython().system_raw("python app.py &")
