# Climate Risk & Insurance Intelligence Dashboard

## Overview

This project is a Streamlit-based dashboard designed to integrate, analyze, and visualize various climate-related data streams and model outputs. The goal is to provide actionable insights relevant to climate risk assessment, potential insurance impacts, adaptation strategies, and broader climate consequences like migration. It leverages real-time weather data, historical local data analysis, simulation techniques, and a multimodal AI assistant powered by Google Gemini.

This dashboard was developed as part of the InnovateHER hiring challenge conducted by Chubb.

## Demo Video

A walkthrough video demonstrating the dashboard's features and functionality (approx. 10 minutes) is available here:

**[https://drive.google.com/file/d/1I2zYkY6whWnDLCFvmCel4_IgBaCu8lh8/view?usp=sharing](https://drive.google.com/file/d/1I2zYkY6whWnDLCFvmCel4_IgBaCu8lh8/view?usp=sharing)**

## Dashboard Preview



<!-- Using HTML table for side-by-side images -->
<table>
  <tr>
    <td><img src="Screenshot 2025-04-03 164450.png" alt="Dashboard View 1" width="100%"></td>
    <td><img src="Screenshot 2025-04-03 164507.png" alt="Dashboard View 2" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><i>Caption: Main dashboard overview showing title and location input.</i></td>
    <td align="center"><i>Caption: Displaying real-time weather, forecast, and risk factors.</i></td>
  </tr>
</table>





## Features
*   **innovate_her_hackathon.py** file contains the core code implementing all functionalities and features.
*   **Real-time Weather & Forecast:** Fetches and displays current conditions and a 5-7 day forecast for any specified city using the OpenWeatherMap API.
*   **Local Temperature Prediction:** Ingests historical weather data (`local_weather.csv`), preprocesses it, and trains a Ridge Regression model (scikit-learn) to predict the next day's maximum temperature. Includes performance visualization and model details.



    <br>
    <div align="center">
      <img src="Screenshot 2025-04-03 164811.png" alt="Local Temperature Prediction Plot" width="75%">
      <br><i>Caption: Plot comparing actual vs. predicted maximum temperatures from the local model.</i>
    </div>



*   **Climate Risk & Insurance Simulation:**
    *   Calculates simple risk factors based on current weather.
    *   Simulates insurance claims and links them to a climate risk index via a basic linear model for projection demonstration.
    *   Provides qualitative insights into potential financial impacts and mitigation strategies relevant to the insurance sector.
*   **Sector & Indicator Analysis:** Allows exploration of descriptions for various climate indicators across different sectors (e.g., Agriculture, Infrastructure, Health).
*   **Climate-Induced Migration Analysis (Simulated):** Visualizes simulated trends, causes, destinations, and future projections of climate migration. Includes risk scoring, correlation analysis, and policy recommendations based on simulated data.
*   **Advanced Climate Model Demo (MetNet - Simplified):** Includes a highly simplified PyTorch implementation demonstrating the *concept* of using advanced models for spatial climate predictions (uses synthetic data, not operational).
*   **Multimodal AI Assistant:** Integrates Google Gemini Pro Vision API for interactive Q&A, capable of understanding both text prompts and uploaded images related to climate risk and adaptation.


    <br>
    <div align="center">
      <img src="Screenshot 2025-04-03 164744.png" alt="Multimodal AI Assistant Interaction" width="75%">
      <br><i>Caption: Example interaction with the AI chatbot analyzing an uploaded image.</i>
    </div>



*   **Interactive UI:** Built with Streamlit, utilizing tabs, selectors, plots, and custom styling for user interaction.

## Technology Stack

*   **Language:** Python 3.x
*   **Core Framework:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **Visualization:** Plotly
*   **Machine Learning:** Scikit-learn (Ridge Regression, MSE)
*   **Deep Learning Demo:** PyTorch
*   **APIs:** OpenWeatherMap, Google Generative AI (Gemini)
*   **Configuration:** python-dotenv
*   **Image Handling:** Pillow

## Project Structure

*   `Final_Dashboard.py`: **(Main Script)** This file contains the primary Python code for the Streamlit application, including the UI layout, data fetching logic, function calls, and integration of all components.
*   `local_weather.csv`: **(Dataset)** This CSV file contains the historical weather data used for training and evaluating the local temperature prediction model. The data for this example was sourced from:
    *   NOAA NCEI Climate Data Online: [https://www.ncei.noaa.gov/cdo-web/search](https://www.ncei.noaa.gov/cdo-web/search)
    *(Note: You need to place your downloaded/prepared CSV with this name in the project's root directory).*
*   `weather_hack.ipynb`: **(Model Development Notebook - Likely)** This Jupyter Notebook likely contains the exploratory data analysis, feature engineering experiments, and model development/tuning steps for the local weather prediction model whose final functions are used within `Final_Dashboard.py`.
*   `requirements.txt`: Lists the necessary Python packages for the project.
*   `.env`: (User must create) File to store API keys securely (see Setup).
*   `README.md`: This file.
*   *Various Screenshot `.png` files*: Located in the root directory, used for display in this README.
*   *(Other support files/folders may be present)*

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher recommended.
    *   `pip` package installer.
    *   Git (for cloning the repository).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kaushika-personal/climate_risk_prediction.git
    cd climate_risk_prediction
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up API Keys:**
    *   You need API keys from:
        *   [OpenWeatherMap](https://openweathermap.org/api) (Free tier is sufficient for basic use)
        *   [Google AI Studio](https://aistudio.google.com/) (For Gemini API key)
    *   Create a file named `.env` in the project's root directory.
    *   Add your keys to the `.env` file like this:
        ```dotenv
        OPENWEATHERMAP_API_KEY=your_openweathermap_api_key_here
        GOOGLE_API_KEY=your_google_gemini_api_key_here
        ```
    *(Note: The Tavily API key mentioned in earlier versions is not required for the final script)*

6.  **Prepare Dataset:**
    *   Ensure the `local_weather.csv` file (sourced as described above) is present in the root directory of the project.

## Running the Application

1.  Make sure your virtual environment is activated.
2.  Ensure the `.env` file and `local_weather.csv` are correctly set up in the project directory.
3.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run innovate_her_hackathon.py
    ```
4.  The dashboard should open automatically in your web browser.

## Data Source Acknowledgment

*   Real-time and forecast weather data is sourced from the [OpenWeatherMap API](https://openweathermap.org/).
*   The historical local weather dataset (`local_weather.csv`) example is based on data available from [NOAA National Centers for Environmental Information (NCEI)](https://www.ncei.noaa.gov/).
*   Climate migration and insurance claims data used in relevant sections are **simulated** for demonstration purposes within the application code.


---
