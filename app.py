import streamlit as st
import pandas as pd
import joblib
import os
import requests
from datetime import datetime,timedelta

st.set_page_config(
    page_title="AI Multi-Hazard Early Warning System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Multi-Hazard Early Warning System")
st.write("AI-based hazard risk analysis dashboard")
# OFFICIAL DISASTER INFORMATION

st.subheader("🛡️ Official Disaster Information")

st.info(
    "This application is a prototype, not an official warning service. "
    "Check government alerts before taking emergency decisions."
)

st.markdown(
    "[🇮🇳 Open NDMA SACHET Official Alert Portal](https://sachet.ndma.gov.in/)"
)

st.markdown(
    "[🌊 CWC River Water-Level Dataset - National Water Data Portal]"
    "(https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc)"
)

st.warning(
    "Educational prototype only. "
    "Not for real emergency decisions."
)
# LIVE WEATHER DATA - FLOOD MODULE

st.header("🌧️ Live Weather Monitoring")

cities = {
    # North India
    "New Delhi": (28.6139, 77.2090),
    "Chandigarh": (30.7333, 76.7794),
    "Dehradun": (30.3165, 78.0322),
    "Shimla": (31.1048, 77.1734),
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Gurugram": (28.4595, 77.0266),
    "Noida": (28.5355, 77.3910),
    "Ghaziabad": (28.6692, 77.4538),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Prayagraj": (25.4358, 81.8463),

    # East India
    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7914, 85.0002),
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Shillong": (25.5788, 91.8933),
    "Agartala": (23.8315, 91.2868),

    # West India
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Panaji": (15.4909, 73.8278),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Raipur": (21.2514, 81.6296),

    # South India
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Tirupati": (13.6288, 79.4192),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kozhikode": (11.2588, 75.7804),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Mysuru": (12.2958, 76.6394),
    "Mangaluru": (12.9141, 74.8560)
}

city = st.selectbox(
    "Select monitoring location",
    list(cities.keys())
)

if st.button("Fetch Live Weather"):
    latitude, longitude = cities[city]

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain"
        ],
        "daily": [
            "precipitation_sum",
            "precipitation_probability_max"
        ],
        "forecast_days": 3,
        "timezone": "auto"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=20
        )
        response.raise_for_status()
        weather = response.json()
       # Check weather data quality
        current = weather.get("current", {})

        required_fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation"
        ]

        missing_fields = [
            field for field in required_fields
            if current.get(field) is None
        ]

        if missing_fields:
            st.warning(
                "Weather data is incomplete. Missing values: "
                + ", ".join(missing_fields)
            )
        else:
            st.success("Weather data quality check passed.")


        st.session_state["live_weather"] = {
            "city": city,
            "data": weather,
            "fetched_at": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            )
        }

    except requests.RequestException as error:
        st.error(f"Weather data fetch failed: {error}")
# Display the most recently fetched weather
if "live_weather" in st.session_state:
    st.caption(
        "Data fetched at: "
        + st.session_state["live_weather"]["fetched_at"]
    )
    saved = st.session_state["live_weather"]
    weather = saved["data"]
    current = weather["current"]

    st.write("Live weather location:", saved["city"])
    st.write(
        "Latest observed/modelled rainfall:",
        current["precipitation"],
        "mm"
    )

    st.subheader(f"Weather for {saved['city']}")

    current = weather["current"]
    daily = weather["daily"]
    if st.button("Use Live Weather for Flood Inputs"):
        st.session_state["rainfall_input"] = float(
            daily["precipitation_sum"][0]
        )
        st.session_state["temperature_input"] = float(
            current["temperature_2m"]
        )
        st.session_state["humidity_input"] = int(
            current["relative_humidity_2m"]
        )
        st.session_state["weather_input_city"] = saved["city"]

        st.success(
            "Weather values selected. Scroll down to review the inputs."
        )
        st.rerun()

    col1, col2 = st.columns(2)

    col1.metric(
        "Current Temperature",
        f"{current['temperature_2m']} °C"
    )

    col2.metric(
        "Current Humidity",
        f"{current['relative_humidity_2m']}%"
    )

    st.write(
        "Current precipitation:",
        f"{current['precipitation']} mm"
    )

    st.write(
        "Current rain:",
        f"{current['rain']} mm"
    )

    forecast_df = pd.DataFrame({
        "Date": daily["time"],
        "Forecast Rainfall (mm)": daily["precipitation_sum"],
        "Maximum Rain Probability (%)":
            daily["precipitation_probability_max"]
    })

    st.subheader("Next 3 Days: Rainfall Forecast")
    st.dataframe(
        forecast_df,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Source: Open-Meteo. Weather-model estimates may differ "
        "from local observations. This data is not a flood warning."
    )
    # HISTORICAL RAINFALL ANALYSIS

st.header("📊 Historical Rainfall Analysis")

historical_city = st.selectbox(
    "Select city for historical rainfall",
    list(cities.keys()),
    key="historical_city"
)

if st.button("Fetch Last 7 Days Rainfall"):
    latitude, longitude = cities[historical_city]

    historical_url = "https://api.open-meteo.com/v1/forecast"

    historical_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "precipitation_sum",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:
        historical_response = requests.get(
            historical_url,
            params=historical_params,
            timeout=20
        )
        historical_response.raise_for_status()

        historical_data = historical_response.json()
        daily_data = historical_data["daily"]

        rainfall_df = pd.DataFrame({
            "Date": daily_data["time"],
            "Rainfall (mm)": daily_data["precipitation_sum"]
        })

        st.subheader(f"Rainfall History: {historical_city}")
        st.dataframe(
            rainfall_df,
            hide_index=True,
            use_container_width=True
        )

        st.line_chart(
            rainfall_df.set_index("Date")["Rainfall (mm)"]
        )

        st.caption(
            "Source: Open-Meteo. Historical weather estimates "
            "are not official flood warnings."
        )

    except (requests.RequestException, KeyError, ValueError) as error:
        st.error(f"Historical rainfall fetch failed: {error}")

# Load flood model
if os.path.exists("flood_model.pkl"):
    model = joblib.load("flood_model.pkl")
else:
    model = None
    # FLOOD MODEL PERFORMANCE

if model is not None and os.path.exists("flood_data.csv"):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    st.subheader("📈 Flood Model Performance")

    data = pd.read_csv("flood_data.csv")

    X = data[["rainfall", "temperature", "humidity"]]
    y = data["risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        
    )

    evaluation_model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100

    st.metric("Test Accuracy (Demo Dataset)", f"{accuracy:.2f}%")

    st.caption(
        "Evaluation uses a small synthetic dataset and a held-out test split. "
        "This score does not measure real-world flood prediction accuracy."
    )



# Store analysis history during the current session
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Hazard selection
hazard = st.selectbox(
    "Select Hazard",
    ["Flood", "Heatwave", "Earthquake", "Landslide"]
)

st.subheader("Enter Environmental Data")

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=1000.0,
    value=20.0,
    key="rainfall_input"
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-50.0,
    max_value=70.0,
    value=30.0,
    key="temperature_input"
)

humidity = st.slider(
    "Humidity (%)",
    min_value=0,
    max_value=100,
    value=60,
    key="humidity_input"
)

magnitude = st.slider(
    "Earthquake magnitude (demo input)",
    0.0, 9.9, 2.0, 0.1
)

slope = st.slider(
    "Slope steepness (degrees)", 0, 90, 15
)

# Analyze button
if st.button("Analyze Hazard Risk"):
    risk = None

    if hazard == "Flood":
        if model is None:
            st.error("Flood model not found. Check project folder.")
        else:
            input_data = pd.DataFrame(
                [[rainfall, temperature, humidity]],
                columns=["rainfall", "temperature", "humidity"]
            )
            risk = str(model.predict(input_data)[0])

    elif hazard == "Heatwave":
        if temperature >= 40:
            risk = "High"
        elif temperature >= 35:
            risk = "Moderate"
        else:
            risk = "Low"

    elif hazard == "Earthquake":
        latitude, longitude = cities[city]

        params = {
            "format": "geojson",
            "latitude": latitude,
            "longitude": longitude,
            "maxradiuskm": 1000,
            "starttime": (
                datetime.utcnow() -
                timedelta(days=30)
            ).strftime("%Y-%m-%d"),
            "minmagnitude": magnitude,
            "orderby": "time",
            "limit": 2000
        }

        try:
            response = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params=params,
                timeout=20
            )
            response.raise_for_status()
            events = response.json().get("features", [])

            records = []

            for event in events:
                p = event.get("properties", {})
                g = event.get("geometry") or {}
                coords = g.get("coordinates") or []

                if p.get("mag") is not None and len(coords) >= 3:
                    records.append({
                        "Magnitude": p["mag"],
                        "Location": p.get("place", "Unknown"),
                        "Depth (km)": coords[2],
                        "Details": p.get("url", "")
                    })

            if not records:
                risk = "Not available"
                st.info("No matching earthquake records found.")

            else:
                earthquake_df = pd.DataFrame(records)
                st.subheader("Recent Earthquake Records")
                st.dataframe(
                    earthquake_df,
                    use_container_width=True,
                    hide_index=True
                )

                max_mag = earthquake_df["Magnitude"].max()

                if max_mag >= 7:
                    risk = "High"
                elif max_mag >= 5:
                    risk = "Moderate"
                else:
                    risk = "Low"

        except requests.RequestException as error:
            risk = None
            st.error(f"USGS data fetch failed: {error}")

    elif hazard == "Landslide":
        if rainfall >= 100 and slope >= 30:
            risk = "High"
        elif rainfall >= 50 and slope >= 20:
            risk = "Moderate"
        else:
            risk = "Low"

    # Show result and save it to history
    if risk is not None:
        st.subheader("Analysis Result")
        st.write("Selected hazard:", hazard)

        if risk == "High":
            st.error("🚨 Demo Risk Category: HIGH")
            st.warning("Alert: High demo risk detected. Review conditions carefully.")

        elif risk == "Moderate":
            st.warning("⚠️ Demo Risk Category: MODERATE")

        elif risk == "Low":
            st.success("✅ Demo Risk Category: LOW")

        else:
            st.info(
                "Earthquake monitoring uses recent USGS records. "
                "It does not predict future earthquakes."
            )

        # Save this analysis
        record = {
            "Hazard": hazard,
            "Rainfall (mm)": rainfall,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Earthquake Magnitude": magnitude,
            "Slope (degrees)": slope,
            "Demo Risk Result": risk
        }

        st.session_state.analysis_history.append(record)

        st.caption(
            "Flood uses a model trained on a small sample dataset. "
            "Heatwave and landslide use illustrative rules. "
            "These results are not official emergency forecasts."
        )

# Input summary
st.subheader("Input Data Summary")

input_summary = pd.DataFrame({
    "Parameter": [
        "Rainfall (mm)",
        "Temperature (°C)",
        "Humidity (%)",
        "Earthquake Magnitude",
        "Slope (degrees)"
    ],
    "Value": [
        rainfall, temperature, humidity, magnitude, slope
    ]
})

st.dataframe(input_summary, hide_index=True)

# Analysis history
st.subheader("📋 Analysis History")

if st.session_state.analysis_history:
    history_df = pd.DataFrame(st.session_state.analysis_history)

    st.dataframe(history_df, hide_index=True, use_container_width=True)

    # Download report
    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=csv_data,
        file_name="hazard_analysis_report.csv",
        mime="text/csv"
    )

    if st.button("Clear Analysis History"):
        st.session_state.analysis_history = []
        st.rerun()

else:
    st.info("No analyses yet. Click Analyze Hazard Risk to create your first record.")