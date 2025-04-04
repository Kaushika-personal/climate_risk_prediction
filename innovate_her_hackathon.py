import streamlit as st
import os
import requests
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch # Still needed for MetNet
import json
import torch.nn.functional as F # Needed for MetNet interpolation
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import io # Needed for image byte handling
from PIL import Image # Needed for image handling

# --- Google Generative AI Imports ---
import google.generativeai as genai

# --- Hugging Face Imports (Removed/Commented Out) ---
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# import accelerate # No longer needed for chatbot

# --- Configuration and Setup ---

# Load environment variables
load_dotenv()

# Tavily key (defined but not used in this script)
tavily_key = os.getenv('TAVILY_API_KEY')

# OpenWeatherMap API key
DEFAULT_WEATHER_KEY = '{{{{{{{{{openweather key}}}}}}}}}}' # Replace with YOUR key if not using .env
weather_api_key = os.getenv('OPENWEATHERMAP_API_KEY', DEFAULT_WEATHER_KEY)

# --- NEW: Google Generative AI API Key ---
# It's strongly recommended to set this in your .env file
# GOOGLE_API_KEY=AIzaSyAjFAHzBGHD4GdkLChXb3ejD26g6xYKBLY
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set page configuration FIRST
st.set_page_config(page_title="Climate Risk & Insurance Intelligence", page_icon="🌍", layout="wide")

# Check for necessary API Keys
api_key_error = False
if not weather_api_key or weather_api_key == 'YOUR_OPENWEATHERMAP_API_KEY':
    st.error("OpenWeatherMap API key not found or default key used. Please set the 'OPENWEATHERMAP_API_KEY' environment variable or replace 'YOUR_OPENWEATHERMAP_API_KEY' in the script.")
    api_key_error = True

if not google_api_key:
    st.error("Google Generative AI API key not found. Please set the 'GOOGLE_API_KEY' environment variable (e.g., in a .env file). Chatbot functionality will be disabled.")
    # Don't stop the app entirely, just disable the chatbot later if key is missing
else:
    # Configure the Google Generative AI client library
    try:
        genai.configure(api_key=google_api_key)
        st.info("Google Generative AI configured successfully.") # Confirmation message
    except Exception as e:
        st.error(f"Error configuring Google Generative AI: {e}")
        google_api_key = None # Disable chatbot if configuration fails

if api_key_error: # Stop if weather key is missing (core functionality)
    st.stop()


# --- Custom CSS ---
def local_css():
    # --- Paste the full CSS from your original code here ---
    st.markdown("""
    <style>
    /* General Body and Font */
    body {
        background-color: #0a192f;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ccd6f6;
    }

    /* Main Title */
    .main-title {
        color: #64ffda;
        font-weight: 700;
        text-align: center;
        margin-top: 20px; /* Add some space at the top */
        margin-bottom: 40px; /* Increased space below title */
    }

    /* Content Cards */
    .result-card {
        background-color: #172a45;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3); /* Slightly stronger shadow */
        padding: 25px; /* Increased padding */
        margin-bottom: 30px; /* Increased space between cards */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px); /* Lift effect */
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }

    /* Metric Cards (within result-card) */
    .metric-card { /* This class seems unused in the final layout, kept for potential future use */
        background-color: #1e3a5f;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 15px;
        margin: 10px 0;
    }

     /* Section Headers */
    h1, h2, h3, h4, h5, h6 {
         color: #e6f1ff; /* Lighter color for headers */
         margin-top: 1.5em;
         margin-bottom: 0.8em;
    }
    h1.main-title { /* Specific override for main title */
         color: #64ffda;
         margin-top: 20px;
         margin-bottom: 40px;
    }
     h2.migration-title {
         color: #64ffda;
         margin-bottom: 20px;
         text-align: center; /* Centered migration title */
     }
     h2.local-pred-title {
         color: #64ffda; /* Match other titles */
         margin-bottom: 20px;
     }


    /* Risk Level Text Colors */
    .risk-high { color: #e74c3c; font-weight: 600; }
    .risk-medium { color: #f39c12; font-weight: 600; }
    .risk-low { color: #2ecc71; font-weight: 600; }

    /* Forecast Day Styling */
    .forecast-day {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        min-height: 310px; /* Use min-height for flexibility */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: background-color 0.3s;
    }
     .forecast-day:hover {
         background-color: #2a4c6f;
     }
    .forecast-day h3 { color: #64ffda; margin-bottom: 10px; font-size: 1.1em; }
    .weather-icon { width: 55px; height: 55px; margin: 8px auto; } /* Slightly larger icon */
    .temperature { font-size: 1.05em; font-weight: bold; color: #f0ab51; margin: 8px 0; }
    .details { font-size: 0.85em; color: #a8b2d1; margin: 4px 0; } /* Slightly lighter details color */
    .details-description { font-size: 0.8em; flex-grow: 1; margin-top: 5px; } /* For description spacing */


    /* Climate Risk (Placeholder Section - Style if used) */
    .climate-risk-title { color: #e03131; font-weight: bold; font-size: 1.2em; margin-bottom: 0.5em; }
    .climate-risk-description { color: #ccd6f6; font-size: 0.9em; line-height: 1.4; }
    .climate-risk-card { background-color: #2a4c6f; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2); }

    /* Indicator Analysis */
    .indicator-description { font-size: 0.95em; color: #a8b2d1; padding-left: 1em; }

    /* Migration Section Specific */
    /* .migration-title is handled above */
    .correlation-card { background-color: #233554; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 3px 7px rgba(0,0,0,0.2); }
    /* Risk level styling for dataframes (applied via pandas styling) */
    .dataframe .risk-high { background-color: rgba(255, 77, 77, 0.7); color: white !important; }
    .dataframe .risk-medium { background-color: rgba(255, 191, 77, 0.7); color: black !important; }
    .dataframe .risk-low { background-color: rgba(77, 255, 77, 0.7); color: black !important; }

    /* Policy Recommendations */
    .recommendation-card { background-color: #233554; border-left: 5px solid #64ffda; padding: 20px; margin: 15px 0; border-radius: 0 8px 8px 0; }
    .financial-impact-item { background-color: #1e3a5f; padding: 15px; margin: 10px 0; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.15); }

    /* Streamlit Widget Styling */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        /* Basic styling, might need !important depending on specificity */
        background-color: #1e3a5f;
        color: #ccd6f6;
        border: 1px solid #395b7c;
        border-radius: 5px;
    }
    .stButton > button {
        background-color: #64ffda;
        color: #0a192f;
        border: none;
        border-radius: 5px;
        padding: 0.6em 1.2em;
        font-weight: 600;
    }
     .stButton > button:hover {
         background-color: #52d6c0;
     }
     /* Chat input related styles */
     .stChatInput { /* Target the container for text input and button if using standard chat */
         /* Styles already defined seem fine, maybe adjust margins */
         margin-top: 10px;
     }
      /* Specific styling for separate text input and button if NOT using st.chat_input */
     .stTextInput > div > div > input { /* Input field */
         background-color: #1e3a5f !important; /* Ensure override */
         color: #ccd6f6 !important;
         border: 1px solid #395b7c !important;
     }
     /* File uploader styling */
     .stFileUploader > div > button { /* Style the button part of the uploader */
          background-color: #233554; /* Slightly different background */
          color: #ccd6f6;
          border: 1px dashed #395b7c;
          padding: 0.5em 1em;
          margin-bottom: 10px; /* Add space below uploader */
     }
     .stFileUploader > div > button:hover {
         background-color: #1e3a5f;
         border-color: #64ffda;
     }


    /* Plotly Chart Background */
    .plotly-chart {
         border-radius: 10px;
         background-color: transparent !important; /* Ensure chart background doesn't override card */
    }
    /* Footer */
     footer {
         text-align: center;
         margin-top: 40px;
         padding-bottom: 20px;
         color: #8892b0; /* Subdued color */
     }
    </style>
    """, unsafe_allow_html=True)

# --- Migration Analysis Functions ---
# [Migration functions: fetch_migration_data, analyze_migration_patterns, create_migration_visualizations, generate_migration_policy_recommendations, migration_section - kept as they are]
def fetch_migration_data():
    """
    Fetch climate-induced migration data.
    For demonstration, using simulated data.
    """
    # Create synthetic migration data
    regions = ["Asia", "Africa", "Europe", "North America", "South America", "Oceania"] # Using Oceania for Australia/Pacific
    migration_data = {}
    current_year = datetime.now().year

    for region in regions:
        # Generate random migration numbers that increase over years
        base_migration = random.randint(50000, 500000)
        yearly_migrations = [int(base_migration * (1 + 0.15 * i + random.uniform(-0.05, 0.05)))
                            for i in range(10)]  # Last 10 years

        # Generate different climate causes with varying percentages
        causes = {
            "Drought": random.randint(20, 40),
            "Flooding": random.randint(15, 35),
            "Sea Level Rise": random.randint(5, 25),
            "Extreme Heat": random.randint(10, 30),
            "Agricultural Failure": random.randint(10, 30),
            "Water Scarcity": random.randint(15, 35)
        }

        # Normalize percentages to sum to 100
        total = sum(causes.values())
        if total == 0: # Avoid division by zero if all random ints are 0
             causes = {k: 100 // len(causes) for k in causes} # Assign equal weight
             remainder = 100 % len(causes)
             keys_list = list(causes.keys())
             for i in range(remainder): # Distribute remainder
                 causes[keys_list[i]] += 1
        else:
            causes = {k: round(v * 100 / total) for k, v in causes.items()} # Use round

            # Ensure they sum to exactly 100% by adjusting the largest category
            diff = 100 - sum(causes.values())
            if diff != 0:
                # Find max key robustly (handles empty dict though it shouldn't happen here)
                max_key = max(causes, key=causes.get) if causes else None
                if max_key:
                    causes[max_key] += diff

        # Top destination countries
        destinations = ["United States", "Germany", "Canada", "Australia", "United Kingdom", "France", "Neighboring Countries"]
        destination_percentages = [random.randint(5, 25) for _ in range(len(destinations))]

        # Normalize destination percentages to sum to 100
        total_dest = sum(destination_percentages)
        if total_dest == 0: # Avoid division by zero
            destination_percentages = [100 // len(destinations)] * len(destinations)
            remainder = 100 % len(destinations)
            for i in range(remainder):
                 destination_percentages[i] += 1
        else:
            destination_percentages = [round(p * 100 / total_dest) for p in destination_percentages] # Use round

            # Ensure they sum to exactly 100%
            diff_dest = 100 - sum(destination_percentages)
            if diff_dest != 0:
                 max_dest_idx = destination_percentages.index(max(destination_percentages)) if destination_percentages else -1
                 if max_dest_idx != -1:
                    destination_percentages[max_dest_idx] += diff_dest

        destination_data = dict(zip(destinations, destination_percentages))

        # Projected future migration based on climate scenarios
        future_projections = {
            "Low Impact (SSP1-2.6)": int(yearly_migrations[-1] * 1.3) if yearly_migrations else 0,
            "Medium Impact (SSP2-4.5)": int(yearly_migrations[-1] * 2.5) if yearly_migrations else 0,
            "High Impact (SSP5-8.5)": int(yearly_migrations[-1] * 4.2) if yearly_migrations else 0
        }

        migration_data[region] = {
            "yearly_data": yearly_migrations,
            "years": list(range(current_year - 10, current_year)), # Add years array
            "causes": causes,
            "destinations": destination_data,
            "future_projections": future_projections
        }

    return migration_data

def analyze_migration_patterns(migration_data):
    """
    Analyze migration patterns.
    Returns risk scores and correlation analysis.
    """
    regions = list(migration_data.keys())
    risk_scores = {}

    for region in regions:
        yearly_data = migration_data[region]["yearly_data"]
        years = np.array(migration_data[region]["years"]).reshape(-1, 1)
        y = np.array(yearly_data)

        # Simple linear regression to get trend
        if len(years) > 1 and len(y) == len(years): # Need at least 2 points for regression and matching lengths
            try:
                model = LinearRegression().fit(years, y)
                trend = model.coef_[0]
            except ValueError: # Handle potential issues if data is unsuitable for regression
                trend = 0
                st.warning(f"Could not calculate migration trend for {region} due to data issues.")
        else:
            trend = 0 # No trend if not enough data points

        # Normalize trend to a 0-100 scale for risk score (Adjust max_trend based on observed data)
        max_trend = 50000  # Adjusted hypothetical maximum trend (can be tuned)
        trend_score = min(100, max(0, (trend / max_trend) * 100)) if max_trend > 0 else 0

        # Create risk score combining trend and the region's climate risk factors
        causes = migration_data[region]["causes"]
        # Weight causes contributing most to displacement
        cause_score = (causes.get("Drought", 0) * 0.25 +
                       causes.get("Flooding", 0) * 0.20 +
                       causes.get("Sea Level Rise", 0) * 0.20 +
                       causes.get("Agricultural Failure", 0) * 0.15 +
                       causes.get("Water Scarcity", 0) * 0.10 +
                       causes.get("Extreme Heat", 0) * 0.10) # Sum of weights = 1.0

        # Final risk score is weighted average of trend and causes
        # Give slightly more weight to the recent trend
        final_score = int(0.6 * trend_score + 0.4 * cause_score)

        risk_scores[region] = {
            "score": final_score,
            "trend": int(trend),
            "level": "High" if final_score > 70 else "Medium" if final_score > 40 else "Low"
        }

    # Calculate correlation between temperature and migration (SIMULATED)
    temp_migration_correlation = 0.76 + random.uniform(-0.05, 0.05)

    # Calculate correlation between extreme weather events and migration (SIMULATED)
    weather_migration_correlation = 0.68 + random.uniform(-0.05, 0.05)

    return {
        "risk_scores": risk_scores,
        "correlations": {
            "temperature_migration": temp_migration_correlation,
            "extreme_weather_migration": weather_migration_correlation
        }
    }

def create_migration_visualizations(migration_data, analysis_results):
    """
    Create visualizations for migration data.
    Returns figures for yearly trends, causes, destinations, and risk map.
    """
    regions = list(migration_data.keys())
    fig_trends = go.Figure()

    # --- Plot Migration Trends ---
    plot_data_available = False
    for region in regions:
        yearly_data = migration_data[region].get("yearly_data", [])
        years = migration_data[region].get("years", [])
        if yearly_data and years and len(yearly_data) == len(years):
            fig_trends.add_trace(
                go.Scatter(
                    x=years,
                    y=yearly_data,
                    mode='lines+markers',
                    name=region
                )
            )
            plot_data_available = True # Mark that we added at least one trace

    if not plot_data_available:
        st.warning("No valid yearly migration data found to plot trends.")
        # Create an empty figure with layout to avoid errors downstream
        fig_trends.update_layout(
            title="Climate-Induced Migration Trends by Region (Data Unavailable)",
            template="plotly_dark", height=450,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
        )
    else:
        fig_trends.update_layout(
            title="Climate-Induced Migration Trends by Region (Past 10 Years)",
            xaxis_title="Year",
            yaxis_title="Estimated Number of Climate Migrants",
            template="plotly_dark",
            height=450, # Adjusted height
            legend_title_text='Region',
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#ccd6f6"
        )

    # --- Plot Causes (Default) ---
    default_region = regions[0] if regions else None
    fig_causes = go.Figure() # Initialize empty
    if default_region and migration_data.get(default_region):
        causes = migration_data[default_region].get("causes", {})
        if causes: # Only plot if causes data exists
            fig_causes.add_trace(
                go.Pie(
                    labels=list(causes.keys()),
                    values=list(causes.values()),
                    hole=.3,
                    pull=[0.05] * len(causes) # Slightly pull slices
                )
            )
            fig_causes.update_layout(
                title=f"Primary Causes of Climate Migration in {default_region}",
                template="plotly_dark", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )
        else: # Handle case where default region has no causes data
            fig_causes.update_layout(
                title=f"Primary Causes in {default_region} (Data Unavailable)",
                template="plotly_dark", height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )
    else: # Handle case where there are no regions
         fig_causes.update_layout(
            title="Primary Causes (Data Unavailable)",
            template="plotly_dark", height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
        )

    # --- Plot Destinations (Default) ---
    fig_destinations = go.Figure() # Initialize empty
    if default_region and migration_data.get(default_region):
        destinations = migration_data[default_region].get("destinations", {})
        if destinations:
            fig_destinations.add_trace(
                go.Bar(
                    x=list(destinations.keys()),
                    y=list(destinations.values()),
                    marker_color=px.colors.qualitative.Pastel,
                    text=[f"{v}%" for v in destinations.values()],
                    textposition='auto'
                )
            )
            fig_destinations.update_layout(
                title=f"Top Destination Areas from {default_region}",
                xaxis_title="Destination Area",
                yaxis_title="Percentage of Migrants (%)",
                template="plotly_dark", height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )
        else:
            fig_destinations.update_layout(
                title=f"Top Destinations from {default_region} (Data Unavailable)",
                template="plotly_dark", height=400,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )
    else:
        fig_destinations.update_layout(
            title="Top Destinations (Data Unavailable)",
            template="plotly_dark", height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
        )

    # --- Plot Risk Map ---
    fig_map = go.Figure() # Initialize empty
    risk_scores = analysis_results.get("risk_scores", {})
    # Map continent names to ISO codes or country names for choropleth
    continent_location_map = {
        "Asia": "China", # Representative large country
        "Africa": "Nigeria", # Representative large country
        "Europe": "Germany",
        "North America": "United States",
        "South America": "Brazil",
        "Oceania": "Australia"
    }

    map_locations = []
    map_z = []
    map_text = []
    if risk_scores:
        for region in regions:
            if region in risk_scores:
                score_data = risk_scores[region]
                map_locations.append(continent_location_map.get(region, region)) # Use mapped location or region name
                score = score_data.get("score", 0)
                level = score_data.get("level", "N/A")
                map_z.append(score)
                map_text.append(f"{region}<br>Risk Score: {score}<br>Level: {level}")

        if map_locations: # Only add trace if we have locations
            fig_map.add_trace(
                go.Choropleth(
                    locations=map_locations,
                    locationmode='country names', # Use country names
                    z=map_z,
                    text=map_text,
                    hoverinfo='text', # Show custom text on hover
                    colorscale="YlOrRd", # Yellow-Orange-Red scale for risk
                    autocolorscale=False,
                    marker_line_color='darkgray',
                    marker_line_width=0.5,
                    colorbar_title="Migration<br>Risk Score",
                    zmin=0,
                    zmax=100 # Assuming score is 0-100
                )
            )
            fig_map.update_layout(
                title_text='Global Climate-Induced Migration Risk Hotspots',
                geo=dict(
                    showframe=False, showcoastlines=True, projection_type='natural earth',
                    bgcolor='rgba(0,0,0,0)', landcolor='rgb(42, 76, 111)', subunitcolor='rgb(200, 200, 200)'
                ),
                template="plotly_dark", height=500, margin={"r":0,"t":40,"l":0,"b":0},
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )
        else: # Handle case where we have risk_scores dict but no valid regions processed
            fig_map.update_layout(
                title_text='Global Risk Hotspots (Data Unavailable)',
                template="plotly_dark", height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
            )

    else: # Handle case where risk_scores dict is empty
         fig_map.update_layout(
            title_text='Global Risk Hotspots (Data Unavailable)',
            template="plotly_dark", height=500,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
        )


    return {
        "trends": fig_trends,
        "causes": fig_causes, # Will be updated dynamically
        "destinations": fig_destinations, # Will be updated dynamically
        "risk_map": fig_map
    }

def generate_migration_policy_recommendations(migration_data, analysis_results):
    """
    Generate policy recommendations based on migration analysis.
    """
    risk_scores = analysis_results.get("risk_scores", {})
    high_risk_regions = [region for region, data in risk_scores.items() if data.get("level") == "High"]
    medium_risk_regions = [region for region, data in risk_scores.items() if data.get("level") == "Medium"]

    common_recommendations = [
        "**Strengthen International Cooperation:** Establish global compacts and funding mechanisms specifically for climate mobility.",
        "**Develop Legal Frameworks:** Advocate for international recognition and protection pathways for climate migrants/refugees.",
        "**Invest in Climate Adaptation:** Increase funding for adaptation projects (e.g., resilient agriculture, water management, coastal defense) in vulnerable source regions.",
        "**Enhance Early Warning Systems:** Link climate forecasts with displacement predictions to enable proactive measures.",
        "**Promote Planned Relocation:** Develop frameworks for voluntary, safe, and dignified relocation for communities facing unavoidable impacts (e.g., sea-level rise)."
    ]

    region_specific_recommendations = {}
    for region, data in risk_scores.items():
        # Ensure region exists in migration_data and has 'causes'
        if region not in migration_data or 'causes' not in migration_data[region]:
            continue # Skip if essential data is missing for the region

        causes = migration_data[region]["causes"]
        risk_level = data.get("level", "Unknown") # Get risk level safely

        # Find top 2 causes safely
        top_causes = sorted(causes.items(), key=lambda item: item[1], reverse=True)[:2] if causes else []
        top_cause_1 = top_causes[0][0] if len(top_causes) > 0 else "Multiple Factors"
        top_cause_2 = top_causes[1][0] if len(top_causes) > 1 else None

        specific_recs = []
        if risk_level == "High":
            specific_recs.append(f"**Priority Action:** Address drivers like {top_cause_1}{f' and {top_cause_2}' if top_cause_2 else ''}.")
            cause_keys = [top_cause_1, top_cause_2] if top_cause_2 else [top_cause_1]
            if any(c in cause_keys for c in ["Drought", "Water Scarcity"]):
                specific_recs.append("Implement large-scale water harvesting, conservation, and efficient irrigation programs.")
            if "Flooding" in cause_keys:
                specific_recs.append("Invest heavily in flood defense infrastructure (levees, drainage) and nature-based solutions (wetland restoration).")
            if "Sea Level Rise" in cause_keys:
                specific_recs.append("Develop and fund coastal zone management plans, including potential managed retreat strategies.")
            if "Agricultural Failure" in cause_keys:
                specific_recs.append("Promote climate-resilient crops, diversification of livelihoods, and access to agricultural insurance.")
            specific_recs.append("Establish dedicated support centers for displaced populations within the region.")

        elif risk_level == "Medium":
            specific_recs.append(f"**Focus Areas:** Build resilience against {top_cause_1}{f' and {top_cause_2}' if top_cause_2 else ''}.")
            specific_recs.append("Strengthen community-based adaptation planning and disaster risk reduction.")
            specific_recs.append("Invest in climate-resilient infrastructure (transport, energy, water).")
            specific_recs.append("Improve climate monitoring and forecasting capabilities at regional/national levels.")

        elif risk_level == "Low": # Explicitly check for "Low"
            specific_recs.append("**Preventative Measures:** Focus on long-term adaptation planning.")
            specific_recs.append("Integrate climate change considerations into national development plans.")
            specific_recs.append("Promote education and awareness about climate impacts and adaptation options.")
        # else: Unknown risk level, add no specific recs

        if specific_recs: # Only add if recommendations were generated
            region_specific_recommendations[region] = specific_recs

    # Generate financial impact assessment (Simulated examples)
    financial_impacts = {
        "Global Economic Output": "Potential reduction of 1-3% annually by 2050 due to displacement and adaptation costs without significant action.",
        "Infrastructure Investment Gap": "Estimated $1-2 trillion needed globally by 2040 for climate-resilient infrastructure in vulnerable areas.",
        "Insurance Sector": "Increased underwriting complexity and potential for correlated losses across regions. Estimated $50-100 billion annual increase in climate-related insured losses by 2035.",
        "Humanitarian Aid": "Projected doubling of aid requirements for climate-related displacement crises by 2040.",
        "Remittances": "Potential disruption to remittance flows, impacting economies in both source and destination countries."
    }

    return {
        "common_recommendations": common_recommendations,
        "region_specific_recommendations": region_specific_recommendations,
        "financial_impacts": financial_impacts
    }

# --- REMOVED: OLD HF FUNCTION ---
# def get_hf_chatbot_response(user_input):
#     # ... (Old implementation removed) ...

# --- REMOVED: Qwen2-VL Helper Function ---
# def process_vision_info_local(messages):
#    # ... (Old implementation removed) ...

# --- REMOVED: Qwen2-VL Chatbot Response Function ---
# def get_qwen_vl_response(user_text, uploaded_image_bytes=None):
#     # ... (Old implementation removed) ...

# --- NEW: Google Gemini Chatbot Response Function ---
def get_gemini_response(user_text, uploaded_image_bytes=None):
    """ Get response from Google Gemini API (gemini-pro-vision)."""

    if not google_api_key:
         return "Chatbot is not available (API key not configured)."

    if not user_text and not uploaded_image_bytes:
        return "Please enter a question or upload an image."

    model = genai.GenerativeModel('gemini-pro-vision')
    contents = []
    pil_image = None

    # Prepare image data if provided
    if uploaded_image_bytes:
        try:
            pil_image = Image.open(io.BytesIO(uploaded_image_bytes))
            # Check image format/mode - Gemini might have preferences, RGB is safe
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            contents.append(pil_image) # Append the PIL image object directly
        except Exception as e:
            st.error(f"❌ Error processing uploaded image for Gemini: {e}")
            # Optionally return error or continue with text only
            # return "Sorry, I couldn't process the uploaded image."

    # Prepare text data if provided
    if user_text:
        contents.append(user_text) # Append the text string

    if not contents: # Should not happen based on initial check, but for safety
        return "No input provided."

    # --- Generate response using Gemini API ---
    try:
        st.write(f"Debug: Sending content to Gemini Vision...")
        response = model.generate_content(contents)
        st.write("Debug: Received response from Gemini.")

        # --- Extract and clean the response ---
        if response and hasattr(response, 'text'):
            output_text = response.text
            # Optionally check for blocks/safety ratings if needed:
            # if response.prompt_feedback.block_reason:
            #     st.warning(f"Response blocked due to: {response.prompt_feedback.block_reason}")
            #     return f"Sorry, your request was blocked ({response.prompt_feedback.block_reason}). Please modify your prompt."
        elif response and response.parts: # Sometimes text is in parts
             output_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            output_text = "Sorry, I couldn't generate a response (empty or unexpected format)."
            st.warning(f"Unexpected Gemini response structure: {response}")


        st.write(f"Debug: Raw output: {output_text[:100]}...")

    except Exception as e:
         st.error(f"❌ Error calling Google Gemini API: {e}")
         # Provide more specific feedback if possible
         if "API key not valid" in str(e):
             return "Sorry, the Google API Key seems invalid. Please check it."
         elif "quota" in str(e).lower():
              return "Sorry, the free usage quota might have been exceeded. Please check your Google AI Studio account."
         # Generic error
         return "Sorry, an error occurred while communicating with the AI assistant."

    return output_text.strip() if output_text else "Generation failed to produce text."


def migration_section():
    st.markdown("<h2 class='migration-title'>Climate-Induced Migration Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    # Fetch and analyze data
    with st.spinner("Fetching and analyzing migration data..."):
        migration_data = fetch_migration_data()
        analysis_results = analyze_migration_patterns(migration_data)

    if not migration_data or not analysis_results:
         st.warning("Could not fetch or analyze migration data. Section may be incomplete.")
         st.markdown("</div>", unsafe_allow_html=True) # Close result-card
         return # Exit section if core data is missing

    # Generate Visualizations and Recommendations
    with st.spinner("Generating migration visualizations and recommendations..."):
        migration_visualizations = create_migration_visualizations(migration_data, analysis_results)
        policy_recommendations = generate_migration_policy_recommendations(migration_data, analysis_results)

    # Create tabs
    tab_titles = ["📈 Trends & Causes", "🗺️ Risk & Hotspots", "🎯 Destinations & Projections", "📜 Policy"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Tab 1: Trends & Causes ---
    with tab1:
        st.plotly_chart(migration_visualizations["trends"], use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True) # Add space

        st.subheader("Regional Analysis: Causes of Migration")
        regions_list = list(migration_data.keys())
        if not regions_list:
            st.info("No regions found in migration data.")
        else:
            col1_cause, col2_cause = st.columns([1, 2])
            with col1_cause:
                # Add unique key to prevent widget state issues if regions change
                selected_region_cause = st.selectbox("Select Region:", regions_list, key="cause_region_select_" + str(len(regions_list)))

            # Update causes chart based on selection
            if selected_region_cause and selected_region_cause in migration_data:
                causes = migration_data[selected_region_cause].get("causes", {})
                if causes:
                    causes_fig = go.Figure(
                        data=[go.Pie(
                            labels=list(causes.keys()), values=list(causes.values()),
                            hole=.3, pull=[0.05] * len(causes), marker_colors=px.colors.qualitative.Set3
                        )]
                    )
                    causes_fig.update_layout(
                        title=f"Primary Causes in {selected_region_cause}",
                        template="plotly_dark", height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6",
                        margin=dict(l=10, r=10, t=50, b=10) # Adjusted margins
                    )
                    with col2_cause:
                        st.plotly_chart(causes_fig, use_container_width=True)
                else:
                    with col2_cause:
                        st.info(f"No cause data available for {selected_region_cause}.")
            else:
                 with col2_cause:
                      st.info("Please select a valid region.")

    # --- Tab 2: Risk Assessment & Hotspots ---
    with tab2:
        st.plotly_chart(migration_visualizations["risk_map"], use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True) # Add space

        st.subheader("Regional Risk Scores")
        risk_scores_data = analysis_results.get("risk_scores", {})
        risk_data = []
        if risk_scores_data:
            for region, data in risk_scores_data.items():
                 trend_val = data.get('trend', 0)
                 risk_data.append({
                    "Region": region,
                    "Risk Score": data.get("score", "N/A"),
                    "Avg. Annual Trend": f"{trend_val:+,} migrants/year", # Add sign
                    "Risk Level": data.get("level", "Unknown")
                })

        if risk_data:
            risk_df = pd.DataFrame(risk_data).set_index("Region")

            # Apply styling based on risk level
            def style_risk_level(df):
                style_df = pd.DataFrame('', index=df.index, columns=df.columns)
                for idx, row in df.iterrows():
                    level = row['Risk Level']
                    if level == "High":
                        style_df.loc[idx, 'Risk Level'] = 'background-color: rgba(231, 76, 60, 0.7); color: white;'
                    elif level == "Medium":
                        style_df.loc[idx, 'Risk Level'] = 'background-color: rgba(243, 156, 18, 0.7); color: black;'
                    elif level == "Low": # Explicitly check Low
                        style_df.loc[idx, 'Risk Level'] = 'background-color: rgba(46, 204, 113, 0.7); color: black;'
                    # else: leave default for Unknown/N/A
                return style_df

            # Display styled DataFrame
            st.dataframe(risk_df.style.apply(style_risk_level, axis=None), use_container_width=True)
        else:
            st.info("No risk score data available to display.")


        st.markdown("---") # Divider
        st.subheader("Correlation Analysis (Simulated)")
        correlations = analysis_results.get('correlations', {})
        corr_temp = correlations.get('temperature_migration', 0) # Default to 0 if missing
        corr_weather = correlations.get('extreme_weather_migration', 0)

        col1_corr, col2_corr = st.columns(2)
        with col1_corr:
             temp_strength = 'strong' if corr_temp > 0.7 else 'moderate' if corr_temp > 0.5 else 'weak'
             temp_color = '#e74c3c' if corr_temp > 0.7 else '#f39c12' if corr_temp > 0.5 else '#2ecc71'
             st.markdown(f"""
            <div class="correlation-card">
                <h5>Temperature-Migration Correlation</h5>
                <p style="font-size: 1.8em; font-weight: bold; color: {temp_color}; margin-bottom: 5px;">
                    {corr_temp:.2f}
                </p>
                <p style="font-size: 0.9em;">Indicates a {temp_strength} positive correlation between rising average temperatures and migration numbers.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2_corr:
            weather_strength = 'strong' if corr_weather > 0.7 else 'moderate' if corr_weather > 0.5 else 'weak'
            weather_color = '#e74c3c' if corr_weather > 0.7 else '#f39c12' if corr_weather > 0.5 else '#2ecc71'
            st.markdown(f"""
            <div class="correlation-card">
                <h5>Extreme Weather-Migration Correlation</h5>
                <p style="font-size: 1.8em; font-weight: bold; color: {weather_color}; margin-bottom: 5px;">
                    {corr_weather:.2f}
                </p>
                <p style="font-size: 0.9em;">Suggests a {weather_strength} positive link between the frequency/intensity of extreme events and migration.</p>
            </div>
            """, unsafe_allow_html=True)


    # --- Tab 3: Destination Analysis & Projections ---
    with tab3:
        st.subheader("Regional Analysis: Destinations & Future Projections")
        regions_list_dest = list(migration_data.keys()) # Get fresh list in case it changed
        if not regions_list_dest:
            st.info("No regions found in migration data.")
        else:
            selected_region_dest = st.selectbox("Select Region:", regions_list_dest, key="dest_region_select_" + str(len(regions_list_dest)))
            st.markdown("<br>", unsafe_allow_html=True) # Add space

            if selected_region_dest and selected_region_dest in migration_data:
                col1_dest, col2_dest = st.columns(2)

                # --- Destinations Chart ---
                with col1_dest:
                    st.markdown("<h6>Top Destination Areas</h6>", unsafe_allow_html=True)
                    destinations = migration_data[selected_region_dest].get("destinations", {})
                    if destinations:
                        dest_fig = go.Figure(
                            data=[go.Bar(
                                x=list(destinations.keys()), y=list(destinations.values()),
                                marker_color=px.colors.qualitative.Pastel,
                                text=[f"{v}%" for v in destinations.values()], textposition='auto'
                            )]
                        )
                        dest_fig.update_layout(
                            xaxis_title=None, yaxis_title="Percentage (%)", template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6",
                            margin=dict(l=10, r=10, t=10, b=10) # Reduced margin
                        )
                        st.plotly_chart(dest_fig, use_container_width=True)
                    else:
                        st.info(f"No destination data available for {selected_region_dest}.")

                # --- Projections Chart ---
                with col2_dest:
                    st.markdown(f"<h6>Future Projections (by 2050, {selected_region_dest})</h6>", unsafe_allow_html=True)
                    projections = migration_data[selected_region_dest].get("future_projections", {})
                    if projections:
                        proj_fig = go.Figure(
                            data=[go.Bar(
                                x=list(projections.keys()), y=list(projections.values()),
                                marker_color=['#2ecc71', '#f39c12', '#e74c3c'], # Green, Orange, Red
                                text=[f"{int(v):,.0f}" for v in projections.values()], # Format number as int
                                textposition='auto'
                            )]
                        )
                        proj_fig.update_layout(
                            xaxis_title="Climate Scenario (SSPs)", yaxis_title="Est. Annual Migrants",
                            template="plotly_dark", height=400,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6",
                            margin=dict(l=10, r=10, t=10, b=10) # Reduced margin
                        )
                        st.plotly_chart(proj_fig, use_container_width=True)
                    else:
                        st.info(f"No future projection data available for {selected_region_dest}.")
            else:
                st.info("Please select a valid region.")


    # --- Tab 4: Policy Recommendations ---
    with tab4:
        st.subheader("Global Policy Recommendations")
        if policy_recommendations.get("common_recommendations"):
            st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
            for rec in policy_recommendations["common_recommendations"]:
                st.markdown(f"▪️ {rec}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No common policy recommendations generated.")

        st.markdown("---") # Divider
        st.subheader("Region-Specific Policy Priorities")
        regions_list_policy = list(policy_recommendations.get("region_specific_recommendations", {}).keys())
        if not regions_list_policy:
            st.info("No region-specific policy recommendations generated.")
        else:
            selected_region_policy = st.selectbox("Select Region:", regions_list_policy, key="policy_region_select_" + str(len(regions_list_policy)))

            if selected_region_policy:
                st.markdown(f'<div class="recommendation-card" style="border-left-color: #f39c12;">', unsafe_allow_html=True) # Orange border
                st.markdown(f"**Priorities for {selected_region_policy}:**")
                region_recs = policy_recommendations["region_specific_recommendations"].get(selected_region_policy, [])
                if region_recs:
                    for rec in region_recs:
                        st.markdown(f"▪️ {rec}")
                else:
                    st.markdown("No specific recommendations found for this region.") # Should not happen if region is in list
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---") # Divider
        st.subheader("Potential Financial & Economic Implications")
        financial_impacts = policy_recommendations.get("financial_impacts", {})
        if financial_impacts:
            for category, impact in financial_impacts.items():
                st.markdown(f"""
                <div class="financial-impact-item">
                    <strong>{category}:</strong> {impact}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No financial impact data generated.")

    st.markdown("</div>", unsafe_allow_html=True) # Close result-card for migration section


# --- MetNet2 Model Definition (Simplified Placeholder - Kept as is) ---
# [MetNet2 class definition - kept as it is]
class MetNet2(torch.nn.Module):
    def __init__(self, forecast_steps=8, input_size=64, num_input_timesteps=6, upsampler_channels=64, lstm_channels=32, encoder_channels=32, center_crop_size=16):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.input_size = input_size
        self.num_input_timesteps = num_input_timesteps
        self.center_crop_size = center_crop_size # Not used in simplified version
        self.input_channels_per_step = 12 # Assuming 12 input features per timestep

        # Simplified layers
        self.encoder = torch.nn.Conv2d(num_input_timesteps * self.input_channels_per_step, encoder_channels, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(encoder_channels, lstm_channels)

        # Calculate intermediate size for ConvTranspose2d input
        self.intermediate_h = input_size // 4
        self.intermediate_w = input_size // 4
        self.fc2_out_features = (upsampler_channels // 4) * self.intermediate_h * self.intermediate_w

        self.fc2 = torch.nn.Linear(lstm_channels, self.fc2_out_features)

        # Channels for ConvTranspose2d input
        self.upsample_in_channels = upsampler_channels // 4

        # Output channels should match input channels per step if predicting same features
        self.output_channels = self.input_channels_per_step

        # ConvTranspose2d to upsample from intermediate size to input_size
        self.upsample_conv = torch.nn.ConvTranspose2d(
            self.upsample_in_channels,
            self.output_channels,
            kernel_size=4, stride=4, padding=0 # Upsamples intermediate_size -> input_size
        )

        self.upsampler_channels = upsampler_channels


    def forward(self, x, lead_time):
        # x shape: [batch, timesteps, channels, height, width]
        # Expected channels = self.input_channels_per_step (12)
        # Expected height = self.input_size (64)
        # Expected width = self.input_size (64)
        # Expected timesteps = self.num_input_timesteps (6)
        batch_size, timesteps, channels, height, width = x.shape

        # Basic input validation
        if timesteps != self.num_input_timesteps or channels != self.input_channels_per_step or \
           height != self.input_size or width != self.input_size:
            st.error(f"MetNet Input Shape Mismatch! Got {x.shape}, Expected: {(batch_size, self.num_input_timesteps, self.input_channels_per_step, self.input_size, self.input_size)}")
            # Return zeros in the expected output shape
            return torch.zeros(batch_size, self.output_channels, height, width, device=x.device)


        x = x.view(batch_size, timesteps * channels, height, width) # Reshape to [B, T*C, H, W]

        x = self.encoder(x) # [B, enc_channels, H, W]
        x = F.relu(x) # Use F.relu

        x = self.pool(x) # [B, enc_channels, 1, 1]
        x = x.view(batch_size, -1) # Flatten [B, enc_channels]
        x = self.fc1(x) # [B, lstm_channels]
        x = F.relu(x) # Use F.relu

        # Simplified lead time incorporation
        lead_time_factor = torch.tensor([(lead_time + 1.0) / self.forecast_steps], device=x.device, dtype=x.dtype)
        x = x * lead_time_factor

        x = self.fc2(x) # [B, fc2_out_features] -> [B, (upsampler_channels/4) * H/4 * W/4]
        x = F.relu(x) # Use F.relu

        # Reshape for ConvTranspose2d
        try:
            # Target shape: [B, upsample_in_channels, intermediate_h, intermediate_w]
            x = x.view(batch_size, self.upsample_in_channels, self.intermediate_h, self.intermediate_w)
        except RuntimeError as e:
            st.error(f"MetNet: Error reshaping tensor before upsampling: {e}. "
                     f"Input shape to view: {x.shape}, Target view: {(batch_size, self.upsample_in_channels, self.intermediate_h, self.intermediate_w)}")
            return torch.zeros(batch_size, self.output_channels, height, width, device=x.device)

        # Upsample
        x = self.upsample_conv(x) # [B, output_channels, H, W]

        # Ensure exact output size (might be needed if stride/padding/kernel don't align perfectly)
        if x.shape[2] != height or x.shape[3] != width:
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)

        # Final shape: [B, output_channels(12), H(64), W(64)]
        return x


# --- Initialization Functions ---

# --- REMOVED: Initialize Hugging Face Model and Tokenizer ---
# @st.cache_resource(...)
# def initialize_hf_model_and_tokenizer(...):
#    # ... (Old implementation removed) ...

# --- REMOVED: Initialize Qwen2-VL Model and Processor ---
# @st.cache_resource(...)
# def initialize_qwen_vl_model_and_processor(...):
#    # ... (Old implementation removed) ...


@st.cache_resource(show_spinner="Initializing MetNet Model...")
def initialize_metnet():
    """Initialize the (simplified) MetNet2 model"""
    try:
        model = MetNet2(
            forecast_steps=8,
            input_size=64, # Match data generation
            num_input_timesteps=6, # Match data generation
            upsampler_channels=64, # Reduced channels
            lstm_channels=32,
            encoder_channels=32, # Reduced channels
            center_crop_size=16,
        )
        model.eval() # Set model to evaluation mode
        st.success("MetNet model initialized.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to initialize MetNet model: {e}")
        return None

# --- Data Fetching & Generation Functions ---
# [Data fetching functions: fetch_real_time_data, fetch_insurance_claims, generate_historical_weather_data, train_risk_model, predict_future_climate, generate_forecast_data - kept as they are]
def fetch_real_time_data(city):
    """
    Fetch real-time data from OpenWeather API.
    Returns a dictionary with weather data or None on failure.
    """
    api_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}&units=metric'
    try:
        response = requests.get(api_url, timeout=10) # 10 second timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()

        # Robust check for expected data structure
        if response.status_code == 200 and isinstance(data, dict) and \
           'main' in data and isinstance(data['main'], dict) and \
           'weather' in data and isinstance(data['weather'], list) and data['weather']:

            main = data['main']
            weather = data['weather'][0]
            wind = data.get('wind', {}) # Use .get for optional keys

            return {
                'temperature': main.get('temp'), # Use .get for safety within dicts
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'wind_speed': wind.get('speed'),
                'location': data.get('name', city),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'weather_description': weather.get('description', 'N/A').capitalize(),
                'icon': weather.get('icon')
            }
        else:
            error_msg = data.get('message', f'Unexpected API response structure or status code {response.status_code}')
            st.error(f"OpenWeatherMap Error for '{city}': {error_msg}")
            return None

    except requests.exceptions.Timeout:
        st.error(f"API request timed out for '{city}'.")
        return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
             st.error(f"API request failed for '{city}': Invalid API Key (401). Check 'weather_api_key'.")
        elif status_code == 404:
             st.error(f"API request failed for '{city}': City not found (404). Check spelling.")
        elif status_code == 429:
             st.error(f"API request failed for '{city}': Rate limit exceeded (429). Please wait.")
        else:
             st.error(f"API request failed for '{city}': HTTP Error {status_code} - {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API network request failed for '{city}': {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Failed to decode API response for '{city}'. Invalid JSON received.")
        return None
    except Exception as e: # Catch any other unexpected errors
        st.error(f"An unexpected error occurred while fetching weather for '{city}': {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed debug info
        return None

def fetch_insurance_claims():
    """
    Fetch real-time insurance claims data. (Simulated)
    Returns 12 months of simulated data.
    """
    # Simulate claims data - base + seasonality + noise
    base_claims = 1000000
    month = datetime.now().month
    # Seasonality peaks around July/August (months 7/8)
    seasonal_variation = np.sin((month - 4) * np.pi / 6) * 200000 # Amplitude 200k
    # Noise standard deviation
    noise_std = 150000

    # Generate 12 random numbers from normal distribution
    claims = np.random.normal(loc=base_claims + seasonal_variation, scale=noise_std, size=12)

    # Ensure claims are non-negative (set a floor)
    claims[claims < 50000] = 50000 # Minimum claim amount $50k

    return claims.astype(int)

def generate_historical_weather_data():
    """Generate synthetic historical weather data for MetNet input (1x6x12x64x64)"""
    try:
        batch_size = 1
        timesteps = 6
        channels = 12
        height = 64
        width = 64
        historical_data = np.zeros((batch_size, timesteps, channels, height, width), dtype=np.float32)

        # Create coordinate grid
        x_coords = np.linspace(-5, 5, width)
        y_coords = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x_coords, y_coords)

        for t in range(timesteps):
            time_factor = t / float(timesteps - 1) if timesteps > 1 else 0 # Normalize time factor

            # Simulate evolving patterns over time steps
            # Temperature (Channel 0): Base + Gradient + Wave + Noise
            temp_base = 15
            temp_gradient = X * 0.5
            temp_wave = 5 * np.sin(X * 0.8 + time_factor * 2 * np.pi) * np.cos(Y * 0.6)
            temp_noise = np.random.normal(0, 1.0, (height, width)) # Increased noise slightly
            historical_data[0, t, 0] = temp_base + temp_gradient + temp_wave + temp_noise

            # Humidity (Channel 1): Base + Different Wave Pattern + Noise
            hum_base = 60
            hum_wave = 15 * np.cos(X * 0.5 - time_factor * 1.5 * np.pi) * np.sin(Y * 0.9 + X * 0.2)
            hum_noise = np.random.normal(0, 5, (height, width)) # Increased noise
            historical_data[0, t, 1] = hum_base + hum_wave + hum_noise

            # Pressure (Channel 2): Moving High/Low Pressure System + Noise
            pres_base = 1013
            center_x = 2.5 * np.cos(time_factor * 1.5 * np.pi) # Adjusted movement
            center_y = 2.5 * np.sin(time_factor * 1.5 * np.pi)
            pres_system = 15 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 10.0) # Adjusted width
            pres_noise = np.random.normal(0, 1.5, (height, width)) # Increased noise
            historical_data[0, t, 2] = pres_base + pres_system + pres_noise

            # Other Channels (3-11): Simpler Patterns or Noise
            for c in range(3, channels):
                 pattern_scale = 2.0
                 noise_scale = 1.5
                 pattern = pattern_scale * np.sin(X / (c + 2.0) + Y / (c + 3.0) + time_factor * (c / 5.0) * np.pi)
                 noise = np.random.normal(0, noise_scale, (height, width))
                 historical_data[0, t, c] = pattern + noise

        # --- Clipping ---
        # Temperature: Clip to a reasonable range (e.g., -30°C to 55°C)
        historical_data[0, :, 0] = np.clip(historical_data[0, :, 0], -30, 55)
        # Humidity: Clip to 0-100%
        historical_data[0, :, 1] = np.clip(historical_data[0, :, 1], 0, 100)
        # Pressure: Clip to a reasonable range (e.g., 940 hPa to 1060 hPa)
        historical_data[0, :, 2] = np.clip(historical_data[0, :, 2], 940, 1060)
        # Clip other channels if necessary based on their expected physical ranges

        return torch.tensor(historical_data, dtype=torch.float32)

    except Exception as e:
        st.error(f"Error generating synthetic historical weather data: {e}")
        return None


def train_risk_model(climate_data, insurance_claims):
    """
    Train a simple linear regression model: Simulated Risk Index -> Simulated Insurance Claims.
    Uses current real-time data to generate *current* risk index, then simulates
    historical risk indices to train the model against simulated historical claims.
    Finally, projects future claims based on a simple extrapolation of the risk index.

    Returns:
        - future_risk_proj.flatten(): Numpy array of 12 projected risk indices.
        - future_claims.astype(int): Numpy array of 12 projected integer claim amounts.
        Returns empty arrays if inputs are invalid or model fails.
    """
    if climate_data is None or not isinstance(climate_data, dict):
        st.warning("Cannot train risk model: Missing or invalid real-time climate data.")
        return np.array([]), np.array([])

    if insurance_claims is None or len(insurance_claims) != 12:
        st.warning(f"Cannot train risk model: Missing or invalid insurance claims data (expected 12 months, got {len(insurance_claims) if insurance_claims is not None else 0}).")
        return np.array([]), np.array([])

    # --- Calculate Current Risk Index ---
    # Use .get with defaults for robustness
    temp = climate_data.get('temperature', 15.0) # Default to 15°C
    humid = climate_data.get('humidity', 60.0)   # Default to 60%
    wind = climate_data.get('wind_speed', 2.0)   # Default to 2 m/s

    # Handle potential None values from .get if API returned nulls
    temp = temp if temp is not None else 15.0
    humid = humid if humid is not None else 60.0
    wind = wind if wind is not None else 2.0

    # Define weights (can be tuned)
    temp_weight = 0.5
    humid_weight = 0.2
    wind_weight = 0.3
    wind_scale_factor = 10 # To bring wind speed influence to a similar scale

    risk_index_current = (temp * temp_weight +
                          humid * humid_weight +
                          (wind * wind_scale_factor) * wind_weight)

    # --- Simulate Historical Risk Index for Training ---
    # Simulate 12 months ending roughly at the current risk index level, with noise and trend.
    sim_risk_std_dev = max(1.0, risk_index_current * 0.15) # Standard deviation relative to current index
    # Generate noise around a base level slightly lower than current
    sim_risk_base = risk_index_current * 0.9
    historical_risk_noise = np.random.normal(loc=0, scale=sim_risk_std_dev, size=12)
    # Add a slight upward trend ending near the current index
    historical_trend = np.linspace(-sim_risk_std_dev * 1.5, 0, 12) # Trend from lower base to current

    historical_risk = sim_risk_base + historical_risk_noise + historical_trend
    historical_risk = np.clip(historical_risk, 0, None) # Ensure risk index is non-negative

    # --- Train Linear Regression Model ---
    model = LinearRegression()
    try:
        # Reshape historical_risk to be a 2D array (n_samples, n_features=1)
        X_train = historical_risk.reshape(-1, 1)
        y_train = insurance_claims # Already a 1D array
        model.fit(X_train, y_train)
    except ValueError as e:
        st.error(f"Error fitting linear regression model: {e}. Check data shapes and content.")
        # st.error(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}") # Debugging
        return np.array([]), np.array([])
    except Exception as e: # Catch other potential fitting errors
         st.error(f"Unexpected error during model fitting: {e}")
         return np.array([]), np.array([])


    # --- Predict Future Claims ---
    # Simple projection: Assume risk index increases slightly (e.g., by 10% over 12 months)
    future_risk_proj = np.linspace(risk_index_current, risk_index_current * 1.1, 12)
    future_risk_proj = np.clip(future_risk_proj, 0, None).reshape(-1, 1) # Ensure >=0 and reshape

    try:
        future_claims = model.predict(future_risk_proj)
        future_claims = np.clip(future_claims, 0, None) # Claims cannot be negative
    except Exception as e:
        st.error(f"Error predicting future claims: {e}")
        return np.array([]), np.array([])

    return future_risk_proj.flatten(), future_claims.astype(int)


def predict_future_climate(metnet_model, historical_data):
    """
    Use the (simplified) MetNet model to predict future climate fields.
    Returns predictions tensor [L, C, H, W] or None on failure.
    """
    if metnet_model is None:
        st.warning("MetNet model is not initialized. Cannot predict future climate.")
        return None
    if historical_data is None:
        st.warning("Historical data not available for MetNet prediction.")
        return None

    # Validate historical data shape (should be [B, T, C, H, W])
    expected_shape = (1, metnet_model.num_input_timesteps, metnet_model.input_channels_per_step, metnet_model.input_size, metnet_model.input_size)
    if historical_data.shape != expected_shape:
        st.error(f"MetNet: Invalid historical data shape. Got {historical_data.shape}, Expected {expected_shape}")
        return None

    predictions = []
    num_forecast_steps = metnet_model.forecast_steps # e.g., 8

    with torch.no_grad(): # Ensure gradients are not computed
        for lead_time in range(num_forecast_steps):
            try:
                # The model's forward pass expects [B, T, C, H, W]
                # Move input data to the same device as the model (if applicable, although MetNet is likely small enough for CPU here)
                if hasattr(metnet_model, 'encoder') and hasattr(metnet_model.encoder, 'weight'):
                     model_device = metnet_model.encoder.weight.device
                     historical_data_device = historical_data.to(model_device)
                else:
                     st.warning("Could not determine MetNet device, assuming CPU for input.")
                     model_device = torch.device("cpu")
                     historical_data_device = historical_data.to(model_device)

                pred_step = metnet_model(historical_data_device, lead_time) # Shape [B, C_out, H, W]

                if pred_step is None: # Handle case where model returns None internally
                    st.error(f"MetNet: Prediction failed internally for lead time {lead_time+1}.")
                    return None # Stop prediction if one step fails severely

                # Ensure the output shape is correct before appending
                expected_output_shape = (1, metnet_model.output_channels, metnet_model.input_size, metnet_model.input_size)
                if pred_step.shape != expected_output_shape:
                     st.error(f"MetNet: Unexpected output shape at lead time {lead_time+1}. Got {pred_step.shape}, Expected {expected_output_shape}")
                     return None

                predictions.append(pred_step.squeeze(0)) # Remove batch dim -> [C_out, H, W]

            except Exception as e:
                st.error(f"Error during MetNet prediction step {lead_time+1}: {e}")
                # import traceback; traceback.print_exc() # Uncomment for detailed debug info in console
                return None # Stop prediction on error

    if not predictions:
        st.warning("MetNet: No successful predictions were generated.")
        return None

    # Stack predictions along a new dimension (lead time): [L, C, H, W]
    try:
        # Ensure all tensors in the list have the same shape before stacking
        if not predictions: return None # Handle empty list case again
        first_shape = predictions[0].shape
        if not all(p.shape == first_shape for p in predictions):
            st.error(f"MetNet: Inconsistent prediction shapes across lead times. Example shapes: {[p.shape for p in predictions[:3]]}")
            return None

        predictions_tensor = torch.stack(predictions, dim=0) # -> [L, C, H, W]
        # Expected final shape: [num_forecast_steps, output_channels, H, W]
        final_expected_shape = (num_forecast_steps, metnet_model.output_channels, metnet_model.input_size, metnet_model.input_size)
        if predictions_tensor.shape != final_expected_shape:
            st.error(f"MetNet: Final stacked tensor shape mismatch. Got {predictions_tensor.shape}, Expected {final_expected_shape}")
            return None

        # Move final tensor to CPU for easier handling/plotting later
        return predictions_tensor.cpu()

    except Exception as e:
        st.error(f"Error stacking MetNet predictions: {e}")
        return None


def generate_forecast_data(city):
    """
    Generate 7-day forecast data using OpenWeatherMap 5-day/3-hour API.
    Aggregates 3-hourly data into daily summaries.
    Returns an ordered dictionary keyed by date string (YYYY-MM-DD) for the next 7 days,
    with values being the forecast dictionary or None if data is unavailable for that day.
    """
    api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}&units=metric'
    today = datetime.now().date()
    api_call_successful = False
    forecast_data_ordered = {} # Use a standard dict first

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        api_call_successful = True

        # Validate main structure
        if 'list' not in data or not isinstance(data['list'], list) or 'city' not in data:
             st.warning(f"Forecast data for '{city}' has unexpected structure (missing 'list' or 'city').")
             data['list'] = [] # Ensure 'list' exists even if empty for downstream logic

        # --- Aggregate data by day ---
        daily_aggregated = {} # Key: date object, Value: list of forecast entries for that day
        for entry in data['list']:
            # Minimal validation for essential keys in each entry
            if not isinstance(entry, dict) or not all(k in entry for k in ['dt', 'main', 'weather']):
                 # st.warning("Skipping forecast entry with incomplete data.") # Can be noisy
                 continue
            if not entry['weather'] or not isinstance(entry['weather'], list) or not isinstance(entry['weather'][0], dict):
                 # st.warning("Skipping forecast entry with invalid 'weather' structure.")
                 continue
            if 'icon' not in entry['weather'][0] or 'description' not in entry['weather'][0]:
                 # st.warning("Skipping forecast entry missing weather icon or description.")
                 continue

            entry_time = datetime.fromtimestamp(entry['dt'])
            entry_date = entry_time.date()

            # Only consider entries from today onwards
            if entry_date >= today:
                if entry_date not in daily_aggregated:
                    daily_aggregated[entry_date] = []
                daily_aggregated[entry_date].append(entry)

        # --- Create forecast for the next 7 days ---
        target_dates = [(today + timedelta(days=i)) for i in range(7)]

        for target_date in target_dates:
            date_str = target_date.strftime("%Y-%m-%d")
            weekday = target_date.strftime("%A") # Full weekday name

            if target_date in daily_aggregated and daily_aggregated[target_date]:
                daily_data = daily_aggregated[target_date]
                try:
                    # Extract data safely using .get
                    temps = [e.get('main', {}).get('temp') for e in daily_data if e.get('main', {}).get('temp') is not None]
                    humidity_vals = [e.get('main', {}).get('humidity') for e in daily_data if e.get('main', {}).get('humidity') is not None]
                    wind_speeds = [e.get('wind', {}).get('speed', 0) for e in daily_data] # Default wind to 0 if missing

                    if not temps: # If no valid temperatures found for the day
                        forecast_data_ordered[date_str] = None
                        continue

                    temp_min = min(temps)
                    temp_max = max(temps)
                    humidity = np.mean(humidity_vals) if humidity_vals else 50 # Default humidity if none found
                    wind_speed = np.mean(wind_speeds)

                    # Get representative weather description/icon (e.g., from midday or first entry)
                    midday_entry = next((e for e in daily_data if 11 <= datetime.fromtimestamp(e['dt']).hour <= 14), daily_data[0])
                    weather_info = midday_entry.get('weather', [{}])[0] # Safely get first weather item
                    weather_description = weather_info.get('description', 'N/A').capitalize()
                    icon = weather_info.get('icon', '01d') # Default icon if missing

                    forecast_data_ordered[date_str] = {
                        'weekday': weekday,
                        'date': date_str, # Include date string for consistency
                        'temperature_min': temp_min,
                        'temperature_max': temp_max,
                        'humidity': int(round(humidity)),
                        'wind_speed': round(wind_speed, 1),
                        'weather_description': weather_description,
                        'icon': icon
                    }
                except Exception as agg_err:
                    st.warning(f"Could not process daily forecast for {date_str}: {agg_err}")
                    forecast_data_ordered[date_str] = None # Mark as None on aggregation error
            else:
                # No data found for this target date in the API response
                forecast_data_ordered[date_str] = None

    # --- Handle API call failures ---
    except requests.exceptions.Timeout:
        st.error(f"API request timed out for forecast ('{city}').")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401: st.error(f"Forecast API request failed: Invalid API Key (401).")
        elif status_code == 404: st.error(f"Forecast API request failed: City '{city}' not found (404).")
        elif status_code == 429: st.error(f"Forecast API request failed: Rate limit exceeded (429).")
        else: st.error(f"Forecast API request failed: HTTP Error {status_code} - {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"API network request failed for forecast ('{city}'): {e}")
    except json.JSONDecodeError:
         st.error(f"Failed to decode API forecast response for '{city}'. Invalid JSON.")
    except Exception as e:
        st.error(f"An unexpected error occurred processing forecast data for '{city}': {e}")
        # import traceback; traceback.print_exc() # For detailed debugging

    # If the API call itself failed, ensure the dictionary reflects no data found
    if not api_call_successful:
        target_dates_str = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        return {date_str: None for date_str in target_dates_str} # Return dict with all Nones

    return forecast_data_ordered

# --- Local Weather Prediction Model Functions ---
# [Local prediction functions: load_and_preprocess_local_weather, train_and_predict_local_weather, plot_local_predictions - kept as they are]
@st.cache_data(show_spinner="Loading historical weather data...") # Cache the loaded and processed data
def load_and_preprocess_local_weather(csv_path="local_weather.csv"):
    """Loads, preprocesses, and engineers features for local weather data."""
    # --- File Loading ---
    if not os.path.exists(csv_path):
        st.error(f"❌ Error: The file '{csv_path}' was not found. Please place it in the same directory as the script or provide the correct path.")
        # Create a dummy empty DataFrame to avoid downstream errors
        expected_cols = ["DATE", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN"] # Define expected columns
        dummy_df = pd.DataFrame(columns=expected_cols)
        dummy_df["DATE"] = pd.to_datetime(dummy_df["DATE"])
        dummy_df = dummy_df.set_index("DATE")
        # Try creating the file with headers if it doesn't exist
        try:
            with open(csv_path, 'w') as f:
                f.write(",".join(expected_cols) + "\n")
            st.info(f"Created empty '{csv_path}' with headers. Please populate it with weather data.")
        except Exception as e:
            st.warning(f"Could not create placeholder file '{csv_path}': {e}")
        return None # Return None explicitly on file not found

    try:
        weather = pd.read_csv(csv_path, index_col="DATE", parse_dates=True) # Parse dates on load
    except Exception as e:
        st.error(f"❌ Error reading CSV file '{csv_path}': {e}")
        return None

    # --- Data Selection and Renaming ---
    # Define required columns
    required_cols = ["PRCP", "TMAX", "TMIN"]
    # Check if required columns exist
    missing_cols = [col for col in required_cols if col not in weather.columns]
    if missing_cols:
        st.error(f"❌ Error: Missing required columns in '{csv_path}': {missing_cols}")
        return None

    # Select core columns + optional snow columns if they exist
    core_cols = required_cols[:] # Create a copy
    optional_cols = ["SNOW", "SNWD"]
    for col in optional_cols:
        if col in weather.columns:
            core_cols.append(col)

    core_weather = weather[core_cols].copy()

    # Rename columns
    rename_map = {"PRCP": "precip", "TMAX": "temp_max", "TMIN": "temp_min"}
    if "SNOW" in core_weather.columns: rename_map["SNOW"] = "snow"
    if "SNWD" in core_weather.columns: rename_map["SNWD"] = "snow_depth"
    core_weather.rename(columns=rename_map, inplace=True)

    # --- Data Cleaning & Preprocessing ---
    # Handle potential non-numeric data and placeholders (like 9999)
    numeric_cols = ["precip", "temp_max", "temp_min"]
    if "snow" in core_weather.columns: numeric_cols.append("snow")
    if "snow_depth" in core_weather.columns: numeric_cols.append("snow_depth")

    for col in numeric_cols:
        # Convert to numeric, coercing errors to NaN
        core_weather[col] = pd.to_numeric(core_weather[col], errors='coerce')
        # Replace common placeholder values with NaN (adjust if your data uses others)
        core_weather[col] = core_weather[col].replace(9999.0, np.nan)
        core_weather[col] = core_weather[col].replace(-9999.0, np.nan)

    # Handle missing values specifically
    # Fill precipitation NaNs with 0 (assuming missing means no precipitation)
    core_weather["precip"] = core_weather["precip"].fillna(0)
    if "snow" in core_weather.columns:
        core_weather["snow"] = core_weather["snow"].fillna(0)
    if "snow_depth" in core_weather.columns:
         # Snow depth might be better forward-filled if sensors fail temporarily
         core_weather["snow_depth"] = core_weather["snow_depth"].fillna(method="ffill").fillna(0) # ffill then 0

    # Forward fill temperature NaNs (assuming temperature doesn't change drastically instantly)
    core_weather[["temp_max", "temp_min"]] = core_weather[["temp_max", "temp_min"]].fillna(method="ffill")

    # Drop any remaining rows with NaNs (e.g., at the very beginning before ffill works)
    initial_rows = len(core_weather)
    core_weather = core_weather.dropna()
    if len(core_weather) < initial_rows:
         st.warning(f"Dropped {initial_rows - len(core_weather)} rows with missing values during preprocessing (likely from the start of the dataset).")

    if core_weather.empty:
        st.error("❌ Weather data is empty after cleaning and preprocessing.")
        return None

    # Ensure correct dtypes after cleaning
    try:
        core_weather[numeric_cols] = core_weather[numeric_cols].astype(float)
    except Exception as e:
        st.error(f"❌ Error converting columns to float after cleaning: {e}")
        return None


    # --- Feature Engineering ---
    # Target: Next day's max temperature
    core_weather["target"] = core_weather["temp_max"].shift(-1)

    # Rolling features (calculate safely)
    window_size = 30
    # Need at least window_size rows to calculate rolling mean properly
    if len(core_weather) < window_size + 1: # +1 for the target shift
        st.warning(f"⚠️ Not enough data ({len(core_weather)} rows) to calculate {window_size}-day rolling features. Skipping feature engineering.")
        # Drop rows where target is NaN due to shift
        core_weather = core_weather.dropna(subset=["target"])
        if core_weather.empty:
            st.error("❌ Data became empty after removing rows for target variable.")
            return None
        # Return without rolling features if insufficient data
        # Make sure only necessary columns are returned
        final_cols = [col for col in ["precip", "temp_max", "temp_min", "target"] if col in core_weather.columns]
        if "snow" in core_weather.columns: final_cols.append("snow")
        if "snow_depth" in core_weather.columns: final_cols.append("snow_depth")
        return core_weather[final_cols]


    # Calculate rolling features only if enough data
    core_weather["month_avg_max"] = core_weather["temp_max"].rolling(window_size).mean()
    core_weather["month_day_max_ratio"] = core_weather["month_avg_max"] / core_weather["temp_max"]
    core_weather["max_min_ratio"] = core_weather["temp_max"] / core_weather["temp_min"]
    core_weather['precip_rolling_sum'] = core_weather['precip'].rolling(window_size).sum()

    # Replace inf introduced by division by zero/small numbers
    core_weather.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaNs created by rolling window or target shift
    initial_rows_fe = len(core_weather)
    core_weather = core_weather.dropna() # Drops rows with NaN in *any* column
    if len(core_weather) < initial_rows_fe:
         st.info(f"ℹ️ Removed {initial_rows_fe - len(core_weather)} rows due to NaNs after feature engineering (rolling windows/target shift).")


    if core_weather.empty:
        st.error("❌ Weather data became empty after feature engineering. Ensure sufficient historical data.")
        return None

    st.success(f"✅ Historical weather data loaded and preprocessed ({len(core_weather)} rows).")
    return core_weather

# @st.cache_resource # Caching the model might be complex if it depends on data split dates
def train_and_predict_local_weather(core_weather_df):
    """Trains Ridge model and makes predictions on the test set."""
    if core_weather_df is None or core_weather_df.empty:
        st.warning("⚠️ Cannot train/predict local model: Invalid input weather data.")
        return None, None, None, None # Return None for error, combined_df, model, predictors

    # --- Define Predictors and Target ---
    # Select features that were successfully created and don't have excessive NaNs
    potential_predictors = [
        "precip", "temp_max", "temp_min",
        "month_avg_max", "month_day_max_ratio", "max_min_ratio", "precip_rolling_sum"
    ]
    # Add optional snow features if they exist
    if "snow" in core_weather_df.columns: potential_predictors.append("snow")
    if "snow_depth" in core_weather_df.columns: potential_predictors.append("snow_depth")

    # Keep only predictors that exist in the dataframe after preprocessing/feature eng.
    predictors = [p for p in potential_predictors if p in core_weather_df.columns]
    target = "target"

    if not predictors:
        st.error("❌ No valid predictor columns found in the processed data. Cannot train model.")
        return None, None, None, None
    if target not in core_weather_df.columns:
         st.error("❌ Target column ('target') not found in the processed data.")
         return None, None, None, None

    st.info(f"ℹ️ Using predictors: {predictors}")

    # --- Train/Test Split ---
    # Simple time-based split (e.g., last 20% or fixed date)
    test_fraction = 0.20
    min_test_size = 30 # Minimum number of samples for testing
    split_index = int(len(core_weather_df) * (1 - test_fraction))

    # Ensure split index is valid and leaves enough data for testing
    if split_index <= 0 or len(core_weather_df) - split_index < min_test_size:
        st.error(f"❌ Not enough data for a reliable train/test split (Total: {len(core_weather_df)}, Required Train > 0, Required Test >= {min_test_size}). Needs more historical data.")
        return None, None, None, predictors # Return predictors even if split fails

    train = core_weather_df.iloc[:split_index]
    test = core_weather_df.iloc[split_index:]

    st.info(f"ℹ️ Training model on {len(train)} samples (until {train.index.max().date()}), Testing on {len(test)} samples (from {test.index.min().date()}).")


    # --- Model Training ---
    # Using Ridge Regression (Linear Regression with L2 regularization)
    # Alpha controls regularization strength (higher alpha = more regularization)
    reg_model = Ridge(alpha=0.1)

    try:
        X_train = train[predictors]
        y_train = train[target]
        # Check for NaNs in training data before fitting
        if X_train.isnull().values.any() or y_train.isnull().values.any():
             st.error("❌ Training data contains NaN values after split. Check preprocessing steps.")
             return None, None, None, predictors

        reg_model.fit(X_train, y_train)
        st.success("✅ Local prediction model trained successfully.")
    except Exception as e:
         st.error(f"❌ Error during model training: {e}")
         return None, None, None, predictors


    # --- Prediction ---
    try:
        X_test = test[predictors]
        y_test_actual = test[target]
         # Check for NaNs in test data before predicting
        if X_test.isnull().values.any():
             st.error("❌ Test data (predictors) contains NaN values before prediction. Check preprocessing steps.")
             # Optionally, try imputing NaNs in X_test if appropriate, or return error
             # X_test = X_test.fillna(X_train.mean()) # Example: Impute with training mean
             # st.warning("Imputed NaN values in test predictors with training mean.")
             return None, None, reg_model, predictors # Return error for now

        predictions = reg_model.predict(X_test)
    except Exception as e:
         st.error(f"❌ Error during model prediction: {e}")
         return None, None, reg_model, predictors # Return model even if prediction fails


    # --- Evaluation ---
    try:
        # Ensure no NaNs in actuals or predictions before calculating error
        if y_test_actual.isnull().any() or pd.isnull(predictions).any():
            st.error("⚠️ Cannot calculate MSE: NaN values found in actuals or predictions.")
            error = None
        else:
            error = mean_squared_error(y_test_actual, predictions)
            st.metric("Test Set Mean Squared Error (MSE)", f"{error:.2f}")
    except Exception as e:
         st.error(f"⚠️ Error calculating Mean Squared Error: {e}")
         error = None # Continue without error metric if calculation fails

    # --- Combine Results for Plotting ---
    try:
        combined = pd.DataFrame({
            "actual": y_test_actual,
            "predictions": predictions
        }, index=test.index)
        # Calculate difference only if both columns are valid numbers
        if pd.api.types.is_numeric_dtype(combined["actual"]) and pd.api.types.is_numeric_dtype(combined["predictions"]):
            combined["difference"] = (combined["actual"] - combined["predictions"]).abs()
        else:
            combined["difference"] = np.nan # Assign NaN if types aren't numeric

    except Exception as e:
        st.error(f"Error combining predictions for display: {e}")
        combined = None


    return error, combined, reg_model, predictors # Return trained model and predictors used

def plot_local_predictions(combined_df):
    """Creates a Plotly chart for actual vs predicted temperatures."""
    if combined_df is None or combined_df.empty:
        st.warning("⚠️ No prediction data available to plot.")
        return None
    # Drop rows where actual or predictions might be NaN before plotting
    plot_df = combined_df.dropna(subset=['actual', 'predictions'])
    if plot_df.empty:
        st.warning("⚠️ No valid (non-NaN) prediction data available to plot.")
        return None


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['actual'],
                             mode='lines', name='Actual Temp Max',
                             line=dict(color='#64ffda', width=2))) # Teal color for actual
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['predictions'],
                             mode='lines', name='Predicted Temp Max',
                             line=dict(color='#f39c12', width=2, dash='dash'))) # Orange dashed for predictions

    fig.update_layout(
        # title="Local Temp Max Prediction: Actual vs. Predicted (Test Set)", # Title added in main app
        xaxis_title="Date",
        yaxis_title="Temperature (°C)", # Adjust if Fahrenheit
        template="plotly_dark",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=30, b=20), # Reduced top margin
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#ccd6f6"
    )
    return fig

# --- Main Application ---
def main():
    # Apply custom CSS
    local_css()

    # --- Page Title ---
    st.markdown("<h1 class='main-title'>Climate Risk & Insurance Intelligence Dashboard</h1>", unsafe_allow_html=True)

    # --- Initialize Models Early (shows spinners/errors upfront) ---
    # NOTE: No AI model initialization needed here anymore, Gemini is API-based.
    #       MetNet is still initialized locally.
    metnet_model = initialize_metnet() # Initialize or get the cached MetNet model


    # --- Location Input ---
    st.subheader("Select Location for Real-time Data")
    # Use session state to remember the city input
    if 'city_input' not in st.session_state:
        st.session_state.city_input = "London" # Default city

    city = st.text_input(
        "Enter City Name:",
        value=st.session_state.city_input,
        key="city_input_widget", # Give it a unique key
        on_change=lambda: st.session_state.update(city_input=st.session_state.city_input_widget), # Update state on change
        label_visibility="collapsed"
    )
    st.divider()


    # --- Current Weather & Forecast ---
    st.header(f"Real-Time Weather Overview for {st.session_state.city_input.capitalize()}")
    # Fetch real-time data based on the current city in session state
    climate_data = fetch_real_time_data(st.session_state.city_input)

    if climate_data:
        # Display Current Weather Card
        st.markdown("<div class='result-card' style='padding-bottom: 15px;'>", unsafe_allow_html=True)
        cw_col1, cw_col2, cw_col3 = st.columns([1.2, 0.8, 2]) # Adjusted column ratios

        temp = climate_data.get('temperature')
        humidity = climate_data.get('humidity')
        icon = climate_data.get('icon')
        desc = climate_data.get('weather_description', 'N/A')
        wind = climate_data.get('wind_speed')
        location = climate_data.get('location', st.session_state.city_input)
        timestamp = climate_data.get('timestamp')
        pressure = climate_data.get('pressure')

        with cw_col1:
            st.metric("Temperature", f"{temp:.1f}°C" if temp is not None else "N/A")
            st.metric("Humidity", f"{humidity}%" if humidity is not None else "N/A")
        with cw_col2:
            if icon:
                icon_url = f"http://openweathermap.org/img/wn/{icon}@2x.png"
                try:
                    # Use Streamlit's columns for better centering if needed, or just st.image
                    st.image(icon_url, width=90, caption=desc if desc != 'N/A' else None)
                except Exception as img_err:
                     st.caption(f"{desc} (icon error: {img_err})")
            else:
                st.caption(desc) # Show description if no icon
            st.caption(f"Wind: {wind:.1f} m/s" if wind is not None else "Wind: N/A")
        with cw_col3:
             st.subheader(f"{desc}")
             st.caption(f"📍 Location: {location} | 🕒 Time: {timestamp if timestamp else 'N/A'}")
             st.caption(f"Pressure: {pressure} hPa" if pressure is not None else "Pressure: N/A")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) # Add space

        # Fetch and Display Forecast Card
        forecast_data_ordered = generate_forecast_data(st.session_state.city_input)
        if forecast_data_ordered:
             st.subheader("🗓️ Weather Forecast (Up to 5-7 Days)")
             # Filter out days with no data before creating columns
             valid_forecast_days = {date_str: data for date_str, data in forecast_data_ordered.items() if data is not None}

             if valid_forecast_days:
                  # Limit columns to avoid excessive width on small screens, maybe wrap?
                  max_cols = 7 # Max forecast days to show in columns
                  valid_dates = list(valid_forecast_days.keys())[:max_cols]
                  cols = st.columns(len(valid_dates))

                  for i, date_str in enumerate(valid_dates):
                      forecast = valid_forecast_days[date_str]
                      with cols[i]:
                          date_formatted = datetime.strptime(date_str, '%Y-%m-%d').strftime('%b %d') # Short format e.g., Jul 15
                          icon_url = f"http://openweathermap.org/img/wn/{forecast.get('icon','01d')}.png" # Default icon
                          day_name = forecast.get('weekday', 'N/A')
                          temp_min = forecast.get('temperature_min')
                          temp_max = forecast.get('temperature_max')
                          humidity = forecast.get('humidity')
                          wind_speed = forecast.get('wind_speed')
                          desc = forecast.get('weather_description', 'N/A')

                          # Format values safely
                          temp_str = f"{temp_min:.0f}° / {temp_max:.0f}°C" if temp_min is not None and temp_max is not None else "N/A"
                          humidity_str = f"{humidity}%" if humidity is not None else "N/A"
                          wind_str = f"{wind_speed:.1f} m/s" if wind_speed is not None else "N/A"


                          st.markdown(f"""
                              <div class="forecast-day">
                                  <h3>{day_name[:3]}</h3>
                                  <p class="details" style="font-size: 0.9em;">{date_formatted}</p>
                                  <img src="{icon_url}" class="weather-icon" alt="{desc}">
                                  <p class="temperature">{temp_str}</p>
                                  <p class="details">💧 {humidity_str}</p>
                                  <p class="details">💨 {wind_str}</p>
                                  <p class="details details-description">{desc}</p>
                              </div>
                              """, unsafe_allow_html=True)
             else:
                  # This case means the API call might have worked but returned no *valid* forecast entries
                  st.info(f"ℹ️ No valid forecast days returned by the API for {st.session_state.city_input.capitalize()}. Free tier provides ~5 days.")

        # else: Error messages handled within generate_forecast_data

    else:
        # fetch_real_time_data already displayed specific errors
        st.error(f"Could not retrieve current weather data for {st.session_state.city_input.capitalize()}. Some dashboard features may be limited.")

    st.divider()


    # --- NEW: Local Weather Prediction Section ---
    st.markdown("<h2 class='local-pred-title'>Local Temperature Prediction (Historical Data Model)</h2>", unsafe_allow_html=True)
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.info("ℹ️ This section uses a Ridge Regression model trained on historical data from `local_weather.csv` to predict the next day's maximum temperature.")

    # Load and process data (cached)
    core_weather_processed = load_and_preprocess_local_weather() # Use default path "local_weather.csv"

    if core_weather_processed is not None and not core_weather_processed.empty:

        # Train model and get predictions
        with st.spinner("Training local model and generating predictions..."):
            mse, combined_preds_df, trained_model, predictors_used = train_and_predict_local_weather(core_weather_processed)

        if combined_preds_df is not None:
            st.subheader("Prediction Results on Test Set")
            prediction_fig = plot_local_predictions(combined_preds_df)
            if prediction_fig:
                st.plotly_chart(prediction_fig, use_container_width=True)
            else:
                st.warning("⚠️ Could not generate prediction plot.")

            # --- Performance & Details Expander ---
            with st.expander("View Model Performance & Details"):
                if mse is not None:
                    st.metric("Test Set Mean Squared Error (MSE)", f"{mse:.2f}")
                else:
                    st.write("Test Set Mean Squared Error (MSE): N/A")

                if trained_model and predictors_used:
                    try:
                        coeffs = pd.Series(trained_model.coef_, index=predictors_used)
                        st.write("**Model Coefficients:**")
                        st.dataframe(coeffs.rename("Coefficient Value"), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display model coefficients: {e}")

                st.write("**Data with Largest Prediction Differences:**")
                if "difference" in combined_preds_df.columns:
                     # Check for NaNs before sorting/displaying
                     diff_df_display = combined_preds_df.dropna(subset=['difference'])
                     st.dataframe(diff_df_display.sort_values("difference", ascending=False).head(10), use_container_width=True)
                else:
                    st.write("Difference column not available.")


        else:
            # Errors handled within train_and_predict_local_weather
            st.warning("⚠️ Could not train the model or generate predictions. Check data and predictor configuration in 'local_weather.csv'.")
    # else: Error displayed in load_and_preprocess_local_weather

    st.markdown("</div>", unsafe_allow_html=True) # Close result-card for prediction section
    st.divider()


    # --- Climate Risk & Insurance Analysis ---
    st.header("Climate Risk & Insurance Impact (Simulated)")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    # --- Data for Simulation ---
    # Fetch simulated historical claims (needs 12 months)
    insurance_claims_hist = fetch_insurance_claims()
    # Use the *real-time* climate data fetched earlier to simulate the risk->claims model
    future_risk_indices, future_claims = train_risk_model(climate_data, insurance_claims_hist)

    # --- Tabs ---
    tab_risk, tab_finance, tab_mitigate = st.tabs(["Risk Factors & Trends", "Financial Impact", "Mitigation Insights"])

    with tab_risk:
        st.subheader("Key Risk Factors (Based on Current Real-time Conditions)")
        if climate_data:
            # Determine risk levels based on current data (ensure data exists)
            temp_val = climate_data.get('temperature')
            humid_val = climate_data.get('humidity')
            wind_val = climate_data.get('wind_speed')

            temp_risk = "High" if temp_val is not None and temp_val > 30 else "Medium" if temp_val is not None and temp_val > 25 else "Low" if temp_val is not None else "N/A"
            humid_risk = "High" if humid_val is not None and humid_val > 80 else "Medium" if humid_val is not None and humid_val > 60 else "Low" if humid_val is not None else "N/A"
            wind_risk = "High" if wind_val is not None and wind_val > 10 else "Medium" if wind_val is not None and wind_val > 5 else "Low" if wind_val is not None else "N/A"

            temp_risk_class = f"risk-{temp_risk.lower()}" if temp_risk != "N/A" else ""
            humid_risk_class = f"risk-{humid_risk.lower()}" if humid_risk != "N/A" else ""
            wind_risk_class = f"risk-{wind_risk.lower()}" if wind_risk != "N/A" else ""

            temp_display = f"{temp_val:.1f}°C" if temp_val is not None else "N/A"
            humid_display = f"{humid_val}%" if humid_val is not None else "N/A"
            wind_display = f"{wind_val:.1f} m/s" if wind_val is not None else "N/A"


            risk_factor_col1, risk_factor_col2, risk_factor_col3 = st.columns(3)
            with risk_factor_col1:
                st.markdown(f"**Temperature Risk:** <span class='{temp_risk_class}'>{temp_risk}</span> ({temp_display})", unsafe_allow_html=True)
            with risk_factor_col2:
                 st.markdown(f"**Humidity Risk:** <span class='{humid_risk_class}'>{humid_risk}</span> ({humid_display})", unsafe_allow_html=True)
            with risk_factor_col3:
                st.markdown(f"**Wind Speed Risk:** <span class='{wind_risk_class}'>{wind_risk}</span> ({wind_display})", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Current real-time weather data unavailable to assess risk factors.")


        st.markdown("---")
        st.subheader("Projected Insurance Claims Trend (Simulated, Next 12 Months)")
        # Check if BOTH future_claims and future_risk_indices were generated successfully
        if future_claims.size > 0 and future_risk_indices.size > 0 and len(future_claims) == 12 and len(future_risk_indices) == 12:
            # Create future month labels starting from next month
            try:
                # Ensure start_date calculation handles month rollovers correctly
                now = datetime.now()
                if now.month == 12:
                    start_date = datetime(now.year + 1, 1, 1)
                else:
                    start_date = datetime(now.year, now.month + 1, 1)

                # Use pandas date_range for robust month generation
                months = pd.date_range(start=start_date, periods=12, freq='MS').strftime('%Y-%m') # MS = Month Start frequency
            except Exception as e:
                 st.warning(f"Could not generate month labels for projection chart: {e}")
                 months = [f"Month {i+1}" for i in range(12)] # Fallback labels


            # Create figure with secondary y-axis
            fig_claims = make_subplots(specs=[[{"secondary_y": True}]])

            # Add projected claims trace
            fig_claims.add_trace(
                go.Scatter(x=months, y=future_claims, name="Projected Claims ($)", line=dict(color='#64ffda')),
                secondary_y=False,
            )
            # Add projected risk index trace
            fig_claims.add_trace(
                go.Scatter(x=months, y=future_risk_indices, name="Projected Risk Index", line=dict(color='#f39c12', dash='dot')),
                secondary_y=True,
            )

            # Add figure title and axis labels
            fig_claims.update_layout(
                title_text="Simulated Claim Projection based on Risk Index Trend",
                template="plotly_dark",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#ccd6f6",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_claims.update_xaxes(title_text="Month")
            fig_claims.update_yaxes(title_text="Estimated Claim Amount ($)", secondary_y=False, showgrid=False) # Hide grid for primary
            fig_claims.update_yaxes(title_text="Simulated Risk Index", secondary_y=True, showgrid=True, gridcolor='rgba(170, 170, 170, 0.3)') # Show grid for secondary


            st.plotly_chart(fig_claims, use_container_width=True)
        else:
            st.warning("⚠️ Could not generate insurance claim projections (missing input data or model training failed).")

    # --- Financial Impact Tab ---
    with tab_finance:
        st.subheader("Estimated Financial Impact")
        st.markdown("""
        *(Note: The following are illustrative examples based on simulated data and general climate trends, not specific portfolio data)*

        *   **Increased Claims Frequency:** Potential for 5-15% higher frequency for weather-sensitive lines (e.g., property, agriculture) in upcoming seasons based on general climate projections.
        *   **Severity Impact:** Average claim costs for events like floods or wildfires may rise by 8-20% due to increased intensity and repair costs.
        *   **Reserve Adequacy:** Recommend stress-testing reserves against plausible climate scenarios (e.g., using IPCC pathways), potentially requiring adjustments of 5-10% for highly exposed portfolios.
        *   **Reinsurance Costs:** Anticipate continued hardening of reinsurance market terms, especially for climate-exposed regions/perils. Factor potential cost increases (10-25%+) into pricing models.
        *   **Investment Risk:** Assess transition risks (stranded assets, policy changes) and physical risks (asset damage, supply chain disruption) within investment portfolios using climate value-at-risk (CVaR) metrics.
        """)
        # Display simulated totals if available
        if future_claims.size > 0:
             total_projected = future_claims.sum()
             avg_monthly_proj = future_claims.mean()
             st.markdown("---") # Divider
             st.markdown("#### Simulated Projections (Next 12 Months):")
             st.markdown(f"<div class='financial-impact-item'><strong>Total Projected Claims Amount:</strong> ${total_projected:,.0f}</div>", unsafe_allow_html=True)
             st.markdown(f"<div class='financial-impact-item'><strong>Average Monthly Projected Claim Amount:</strong> ${avg_monthly_proj:,.0f}</div>", unsafe_allow_html=True)
        # else: No claims data to show


    # --- Mitigation Tab ---
    with tab_mitigate:
        st.subheader("Potential Mitigation & Adaptation Strategies (Insurance Context)")
        st.markdown("""
        *   **Enhanced Underwriting & Pricing:** Integrate forward-looking climate risk scores (e.g., from catastrophe models, climate service providers) into risk selection, pricing tiers, and exposure management systems. Move beyond purely historical data.
        *   **Parametric Solutions:** Develop and expand use of index-based insurance triggered by verifiable metrics (wind speed thresholds, rainfall levels, temperature anomalies, modeled loss indices). Beneficial for rapid payouts and covering events where traditional loss adjustment is difficult (e.g., agriculture, some business interruption).
        *   **Policyholder Engagement & Incentives:** Offer premium discounts, deductible reductions, or non-monetary support (e.g., resilience audits) for policyholders implementing verified resilience measures (e.g., FORTIFIED home standards, flood barriers, fire-resistant landscaping, drought-tolerant practices).
        *   **Portfolio Optimization & Diversification:** Analyze geographic and peril concentrations under various climate change scenarios (e.g., RCPs/SSPs). Use insights to optimize reinsurance structures, explore geographic diversification, or adjust underwriting appetite in high-risk zones.
        *   **Community Resilience Partnerships:** Collaborate with municipalities, NGOs, and developers on initiatives like advocating for stronger building codes, supporting community-based early warning systems, investing in green infrastructure (e.g., mangrove restoration, permeable surfaces), and engaging in discussions around managed retreat or land use planning.
        *   **Product Innovation:** Develop new insurance products or riders addressing emerging climate risks, such as:
            *   Heat stress impacts on agriculture yields or worker productivity.
            *   Water scarcity risks for industrial or agricultural users.
            *   Coverage for transition risks associated with decarbonization policies.
            *   Microinsurance products tailored for vulnerable communities.
        *   **Claims Process Adaptation:** Prepare claims teams for increased frequency and severity, potentially utilizing technology like drones and AI for faster damage assessment after large-scale events.
        """)

    st.markdown("</div>", unsafe_allow_html=True) # Close result-card for Risk/Insurance
    st.divider()


    # --- Sector & Indicator Analysis ---
    st.header("Sector & Indicator Deep Dive")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    sector_indicators = { # Keep this data definition
        "Terrestrial Climate": ["Annual maximum temperature", "Annual minimum of monthly mean soil moisture content", "Fire Weather Index during the fire season", "Length of the fire season", "Number of days with extreme fire weather"],
        "Urban Heat Stress": ["Average Daily Temperature", "Number of Heatwave Days", "Urban Heat Island Intensity"],
        "Marine Climate": ["Sea Surface Temperature", "Ocean Acidification", "Sea Level Rise"],
        "Agriculture & Food Security": ["Crop Yield Variability", "Drought Index", "Growing Season Length"],
        "Water Resources": ["River Discharge Anomalies", "Groundwater Depletion Rate", "Water Stress Index"],
        "Biodiversity": ["Species Habitat Shift", "Ecosystem Resilience Index", "Coral Bleaching Events"],
        "Infrastructure": ["Climate Impact on Road Networks", "Energy Grid Vulnerability", "Coastal Infrastructure Risk Index"],
        "Human Health": ["Heat-Related Illness Rates", "Vector-Borne Disease Incidence", "Air Quality Impact Days"],
    }
    indicator_descriptions = { # Keep this data definition
        "Annual maximum temperature": "Highest recorded temperature in a year, indicating heat extremes.",
        "Annual minimum of monthly mean soil moisture content": "Lowest monthly average soil water content, critical for agriculture and ecosystems.",
        "Fire Weather Index during the fire season": "Meteorological index assessing fire danger based on temperature, humidity, wind, and rainfall.",
        "Length of the fire season": "Duration of the period with high fire risk conditions.",
        "Number of days with extreme fire weather": "Count of days where fire danger indices exceed critical thresholds.",
        "Average Daily Temperature": "Mean temperature over a 24-hour period in urban areas.",
        "Number of Heatwave Days": "Count of days where temperatures exceed specific heatwave criteria.",
        "Urban Heat Island Intensity": "Difference in temperature between urban areas and surrounding rural areas.",
        "Sea Surface Temperature": "Temperature of the ocean's surface layer, affecting marine ecosystems and weather patterns.",
        "Ocean Acidification": "Decrease in ocean pH due to CO2 absorption, impacting marine life.",
        "Sea Level Rise": "Increase in the average global sea level, threatening coastal areas.",
        "Crop Yield Variability": "Fluctuations in agricultural output, often linked to weather extremes.",
        "Drought Index": "Standardized measure of drought severity (e.g., SPI, PDSI).",
        "Growing Season Length": "Duration of the period suitable for crop growth.",
        "River Discharge Anomalies": "Deviations from the normal flow rate of rivers, indicating flood or drought.",
        "Groundwater Depletion Rate": "Rate at which groundwater levels are declining.",
        "Water Stress Index": "Ratio of water withdrawal to water availability.",
        "Species Habitat Shift": "Changes in the geographical range of species due to climate change.",
        "Ecosystem Resilience Index": "Measure of an ecosystem's ability to withstand and recover from disturbances.",
        "Coral Bleaching Events": "Occurrences where corals expel algae due to stress (often heat), indicating reef health decline.",
         "Climate Impact on Road Networks": "Assessment of risks like flooding, landslides, and extreme heat affecting roads.",
        "Energy Grid Vulnerability": "Risk assessment of power generation and distribution infrastructure to climate extremes.",
        "Coastal Infrastructure Risk Index": "Combined measure of exposure and vulnerability of coastal assets to sea-level rise and storm surge.",
        "Heat-Related Illness Rates": "Incidence of health issues like heat stroke and exhaustion.",
        "Vector-Borne Disease Incidence": "Frequency of diseases spread by vectors (e.g., mosquitoes, ticks) whose range is affected by climate.",
        "Air Quality Impact Days": "Number of days where air quality standards are exceeded, often exacerbated by heat and wildfires."
    }

    sector_list = list(sector_indicators.keys())
    selected_sector = st.selectbox("Select Sector for Indicator Analysis:", sector_list, key="sector_select")

    if selected_sector:
        indicators = sector_indicators.get(selected_sector, [])
        if indicators:
            # Use a unique key for the indicator selectbox based on the selected sector
            indicator_key = f"indicator_select_{selected_sector.replace(' ', '_')}"
            selected_indicator = st.selectbox("Select Indicator:", indicators, key=indicator_key)
            if selected_indicator:
                description = indicator_descriptions.get(selected_indicator, "No description available.")
                st.markdown(f"**Description of __{selected_indicator}__:**")
                st.markdown(f"<p class='indicator-description'>{description}</p>", unsafe_allow_html=True)
            # else: User has not selected an indicator yet
        else:
            st.info("ℹ️ No specific indicators listed for this sector yet.")
    # else: User has not selected a sector yet

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()


    # --- Climate-Induced Migration Section ---
    migration_section() # Call the existing function
    st.divider()


    # --- MetNet2 Placeholder Prediction Visualization ---
    st.header("Advanced Climate Model Insights (MetNet2 - Simplified Demo)")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.info("ℹ️ Displaying simulated output from a simplified MetNet-like model. This shows potential future spatial patterns based on **synthetic input data** and is **not an operational forecast**.")

    if metnet_model: # Check if model initialized successfully
        # Generate fresh synthetic data each time for the demo
        historical_metnet_data = generate_historical_weather_data()

        if historical_metnet_data is not None:
            with st.spinner("Running simplified MetNet prediction..."):
                climate_predictions = predict_future_climate(metnet_model, historical_metnet_data)

            if climate_predictions is not None:
                # Visualize one channel (e.g., Temperature - Channel 0) for a few lead times
                num_predictions = climate_predictions.shape[0] # Number of lead times available
                num_channels = climate_predictions.shape[1]
                channel_to_plot = 0 # Plot Temperature (index 0)
                channel_name = "Temperature" # Label for the plot

                if num_channels <= channel_to_plot:
                    st.warning(f"⚠️ Cannot plot channel {channel_to_plot}. Model output only has {num_channels} channels.")
                else:
                    # Show predictions for lead time 1, middle, and last
                    lead_times_indices_to_show = [0] # First prediction (+1 step)
                    if num_predictions > 1:
                        lead_times_indices_to_show.append(num_predictions // 2) # Middle prediction
                    if num_predictions > 2:
                        lead_times_indices_to_show.append(num_predictions - 1) # Last prediction
                    lead_times_indices_to_show = sorted(list(set(lead_times_indices_to_show))) # Remove duplicates if num_predictions <= 2

                    pred_cols = st.columns(len(lead_times_indices_to_show))
                    for i, lead_time_idx in enumerate(lead_times_indices_to_show):
                        with pred_cols[i]:
                            st.subheader(f"Lead Time: +{lead_time_idx+1} units")
                            try:
                                # Extract the specific channel prediction (already on CPU from predict func)
                                pred_map = climate_predictions[lead_time_idx, channel_to_plot, :, :].numpy()

                                # Check for NaN/inf in prediction data before plotting
                                if np.isnan(pred_map).any() or np.isinf(pred_map).any():
                                     st.warning(f"⚠️ NaN or Inf values found in MetNet prediction for lead time +{lead_time_idx+1}. Displaying placeholder.")
                                     st.markdown("<div style='height: 250px; background-color: #2a4c6f; display: flex; align-items: center; justify-content: center; border-radius: 8px;'><p>Invalid Data</p></div>", unsafe_allow_html=True)

                                else:
                                    # Determine reasonable color range (can be fixed or dynamic)
                                    vmin, vmax = np.percentile(pred_map, [5, 95]) # Use percentiles for robustness
                                    # Handle cases where vmin/vmax are identical or NaN
                                    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
                                        vmin = np.nanmin(pred_map) # Fallback to actual min/max ignoring NaNs
                                        vmax = np.nanmax(pred_map)
                                        if vmin == vmax: # If still identical (constant field)
                                            vmin = vmin - 0.5 if vmin is not None else -0.5
                                            vmax = vmax + 0.5 if vmax is not None else 0.5
                                        # Handle case where all data might be NaN
                                        if np.isnan(vmin) or np.isnan(vmax):
                                            vmin, vmax = 0, 1 # Absolute fallback

                                    fig_pred = px.imshow(pred_map, color_continuous_scale='RdBu_r', aspect='equal', zmin=vmin, zmax=vmax)
                                    fig_pred.update_layout(
                                        title=f"Simulated {channel_name} Pattern",
                                        coloraxis_colorbar=dict(title="Value"),
                                        margin=dict(l=0, r=0, t=40, b=0), # Adjusted top margin
                                        height=250,
                                        template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#ccd6f6"
                                        )
                                    fig_pred.update_xaxes(showticklabels=False)
                                    fig_pred.update_yaxes(showticklabels=False)
                                    # Ensure unique key using index 'i' and lead_time_idx
                                    st.plotly_chart(fig_pred, use_container_width=True, key=f"metnet_pred_{i}_{lead_time_idx}")
                            except IndexError:
                                st.error(f"❌ Error accessing prediction data for lead time +{lead_time_idx+1}. Index out of bounds.")
                            except Exception as plot_err:
                                 st.error(f"❌ Error plotting MetNet prediction: {plot_err}")
            # else: predict_future_climate already showed error/warning
        # else: generate_historical_weather_data showed error
    else:
        st.warning("⚠️ MetNet model was not initialized successfully. Prediction visualization skipped.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()


    # --- Chatbot (Using Google Gemini API) ---
    st.header("Multimodal AI Assistant (Powered by Google Gemini)")
    st.markdown("<div class='result-card' style='margin-bottom: 20px;'>", unsafe_allow_html=True)

    if not google_api_key:
        # Error message already displayed during initial configuration check
        st.error("🔴 Google AI Assistant is disabled. Please configure the GOOGLE_API_KEY.")
    else:
        # Display info about the model being used
        model_used_name = "gemini-pro-vision" # The model we are using
        st.info(f"ℹ️ Using Google Gemini API (`{model_used_name}`). You can ask questions about uploaded images or general topics.")

        # Initialize chat history (using a generic name now)
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [{"role": "assistant", "content": f"Hi! Ask me a question. You can also upload an image and ask about it using the Google Gemini model."}]

        # Display chat messages
        chat_container = st.container(height=400) # Use a container with fixed height for scroll
        with chat_container:
             for message in st.session_state.chat_messages:
                 with st.chat_message(message["role"]):
                     # Display text content
                     st.markdown(message["content"])
                     # Display image if it exists in the message (for user messages)
                     if message["role"] == "user" and "image_display" in message and message["image_display"] is not None:
                        st.image(message["image_display"], width=150) # Display small image in chat


        # User input using separate widgets for text and image
        uploaded_file = st.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"], key="gemini_uploader")
        prompt = st.text_input("Ask the AI assistant:", key="gemini_prompt_input")
        submit_button = st.button("Send", key="gemini_send_button")


        if submit_button and (prompt or uploaded_file):
            user_message = {"role": "user", "content": prompt if prompt else ""} # Start with text
            image_bytes = None
            image_for_display = None # For showing in chat history

            if uploaded_file is not None:
                 image_bytes = uploaded_file.getvalue()
                 image_for_display = image_bytes # Keep bytes for display
                 # Add a placeholder or indicator text if prompt is empty but image exists
                 if not prompt:
                      user_message["content"] = "[Image Query]" # Indicate it's about the image
                 # Store image for potential display in history
                 user_message["image_display"] = image_for_display


            # Add user message to history and display it immediately
            st.session_state.chat_messages.append(user_message)

            # Rerun is generally not needed here, spinner handles the wait time.
            # Display the user message in the next pass implicitly.

            # Get and display assistant response
            with chat_container: # Draw inside the container
                 with st.chat_message("assistant"): # Use assistant role
                     message_placeholder = st.empty() # Placeholder for the response
                     with st.spinner("🤖 Asking Google Gemini..."):
                         # *** Use the NEW Google Gemini function ***
                         response = get_gemini_response(prompt, image_bytes) # Call the updated function
                         message_placeholder.markdown(response) # Display the full response

            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

            # Clear input fields *after* processing by letting Streamlit rerun naturally
            # Or clear manually if preferred:
            # st.session_state.gemini_prompt_input = ""
            # The file uploader state is harder to reset programmatically without rerunning fully
            # or more complex state management. Usually, the user uploading a new file handles it.

            # Force a rerun to clear the input text box state reliably after sending
            st.rerun()


    st.markdown("</div>", unsafe_allow_html=True) # Close result-card for chatbot


    # --- Resources ---
    st.header("Resources")
    st.markdown("""
    *   [IPCC Reports](https://www.ipcc.ch/) - Intergovernmental Panel on Climate Change assessments.
    *   [NASA Climate Change](https://climate.nasa.gov/) - Data, visualizations, and news from NASA.
    *   [NOAA Climate.gov](https://www.climate.gov/) - U.S. climate science and information.
    *   [OpenWeatherMap](https://openweathermap.org/) - Weather data API provider.
    *   [Climate Adaptation Knowledge Exchange (CAKE)](https://www.cakex.org/) - Case studies and resources.
    *   [Google AI Studio](https://aistudio.google.com/) - Manage Gemini API keys and usage.
    """, unsafe_allow_html=True) # Updated resource


    # --- Footer ---
    st.markdown("---")
    st.markdown("<footer>Climate Risk Intelligence Dashboard © 2024 | Data sources: OpenWeatherMap, Simulated Data, local_weather.csv | Chatbot: Google Gemini API</footer>", unsafe_allow_html=True) # Updated footer


# --- Run the App ---
if __name__ == "__main__":
    # --- Dependency Check --- (Updated)
    missing_libs = []
    try: import torch # Still needed for MetNet
    except ImportError: missing_libs.append("torch")
    # REMOVED: transformers, accelerate, sentencepiece
    # try: import transformers
    # except ImportError: missing_libs.append("transformers")
    # try: import accelerate
    # except ImportError: missing_libs.append("accelerate")
    # try: import sentencepiece # Often needed by tokenizers/processors
    # except ImportError: missing_libs.append("sentencepiece")
    try: import google.generativeai # NEW
    except ImportError: missing_libs.append("google-generativeai")
    try: import sklearn
    except ImportError: missing_libs.append("scikit-learn")
    try: import dotenv
    except ImportError: missing_libs.append("python-dotenv")
    try: import plotly
    except ImportError: missing_libs.append("plotly")
    try: import pandas
    except ImportError: missing_libs.append("pandas")
    try: import PIL # For image handling
    except ImportError: missing_libs.append("Pillow")


    if missing_libs:
        st.error(f"❌ Missing essential libraries: {', '.join(missing_libs)}. Please install all required packages.")
        st.code("pip install streamlit pandas numpy plotly requests python-dotenv Pillow torch scikit-learn google-generativeai") # Updated command
        st.stop()

    main()
