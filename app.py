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

st.header("🌍 Worldwide Live Weather Monitoring")

weather_location = st.text_input(
    "Enter Any City or Country",
    placeholder="Example: Tokyo, Japan"
)

if st.button("Fetch Worldwide Live Weather"):

    if not weather_location.strip():
        st.warning("Please enter a location.")

    else:
        try:
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"

            geo_params = {
                "name": weather_location,
                "count": 10,
                "language": "en",
                "format": "json"
            }

            geo_response = requests.get(
                geo_url,
                params=geo_params,
                timeout=20
            )

            geo_response.raise_for_status()

            locations = geo_response.json().get("results", [])

            if not locations:
                st.warning("Location not found.")

            else:
                selected = locations[0]

                latitude = selected["latitude"]
                longitude = selected["longitude"]

                city_name = selected["name"]
                country = selected.get("country", "Unknown")

                weather_url = (
                    "https://api.open-meteo.com/v1/forecast"
                )

                weather_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "precipitation",
                        "rain",
                        "wind_speed_10m"
                    ],
                    "daily": [
                        "precipitation_sum",
                        "precipitation_probability_max"
                    ],
                    "forecast_days": 3,
                    "timezone": "auto"
                }

                weather_response = requests.get(
                    weather_url,
                    params=weather_params,
                    timeout=20
                )

                weather_response.raise_for_status()

                weather_data = weather_response.json()

                st.session_state["global_weather"] = {
                    "city": city_name,
                    "country": country,
                    "latitude": latitude,
                    "longitude": longitude,
                    "data": weather_data
                }

                st.success("Worldwide weather fetched successfully!")

        except requests.RequestException as error:
            st.error(f"Weather API Error: {error}")


# DISPLAY WEATHER

if "global_weather" in st.session_state:

    saved = st.session_state["global_weather"]

    current = saved["data"]["current"]
    daily = saved["data"]["daily"]

    st.subheader(
        f"Weather: {saved['city']}, {saved['country']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Temperature",
        f"{current['temperature_2m']} °C"
    )

    col2.metric(
        "Humidity",
        f"{current['relative_humidity_2m']}%"
    )

    col3.metric(
        "Rainfall",
        f"{current['precipitation']} mm"
    )

    col4.metric(
        "Wind Speed",
        f"{current['wind_speed_10m']} km/h"
    )

    st.write("Latitude:", saved["latitude"])
    st.write("Longitude:", saved["longitude"])

    forecast_df = pd.DataFrame({
        "Date": daily["time"],
        "Rainfall (mm)": daily["precipitation_sum"],
        "Rain Probability (%)":
            daily["precipitation_probability_max"]
    })

    st.subheader("3-Day Weather Forecast")

    st.dataframe(
        forecast_df,
        hide_index=True,
        use_container_width=True
    )

    if st.button("Use Weather Data for Flood Analysis"):

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

        st.success("Weather data transferred to flood inputs.")

    st.caption(
        "Source: Open-Meteo. Weather data is model-based "
        "and is not an official disaster warning."
    )
    # Worldwide City Coordinates
# Used for Historical Rainfall and Earthquake Monitoring

cities = {
    # India
    "New Delhi": (28.6139, 77.2090),
    "Greater Noida": (28.4744, 77.5040),
    "Noida": (28.5355, 77.3910),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Bengaluru": (12.9716, 77.5946),
    "Mysuru": (12.2958, 76.6394),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Hyderabad": (17.3850, 78.4867),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Kolkata": (22.5726, 88.3639),
    "Siliguri": (26.7271, 88.3953),
    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7955, 84.9994),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Prayagraj": (25.4358, 81.8463),
    "Agra": (27.1767, 78.0081),
    "Meerut": (28.9845, 77.7064),
    "Gorakhpur": (26.7606, 83.3732),
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Raipur": (21.2514, 81.6296),
    "Ranchi": (23.3441, 85.3096),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Shillong": (25.5788, 91.8933),
    "Imphal": (24.8170, 93.9368),
    "Dehradun": (30.3165, 78.0322),
    "Haridwar": (29.9457, 78.1642),
    "Shimla": (31.1048, 77.1734),
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Chandigarh": (30.7333, 76.7794),
    "Panaji": (15.4909, 73.8278),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),

    # Asia
    "Tokyo": (35.6762, 139.6503),
    "Osaka": (34.6937, 135.5023),
    "Seoul": (37.5665, 126.9780),
    "Beijing": (39.9042, 116.4074),
    "Shanghai": (31.2304, 121.4737),
    "Hong Kong": (22.3193, 114.1694),
    "Singapore": (1.3521, 103.8198),
    "Bangkok": (13.7563, 100.5018),
    "Jakarta": (-6.2088, 106.8456),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Manila": (14.5995, 120.9842),
    "Kathmandu": (27.7172, 85.3240),
    "Dhaka": (23.8103, 90.4125),
    "Colombo": (6.9271, 79.8612),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Dubai": (25.2048, 55.2708),
    "Abu Dhabi": (24.4539, 54.3773),
    "Riyadh": (24.7136, 46.6753),
    "Doha": (25.2854, 51.5310),
    "Istanbul": (41.0082, 28.9784),
    "Tashkent": (41.2995, 69.2401),

    # Europe
    "London": (51.5072, -0.1276),
    "Paris": (48.8566, 2.3522),
    "Berlin": (52.5200, 13.4050),
    "Rome": (41.9028, 12.4964),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3874, 2.1686),
    "Amsterdam": (52.3676, 4.9041),
    "Brussels": (50.8503, 4.3517),
    "Zurich": (47.3769, 8.5417),
    "Vienna": (48.2082, 16.3738),
    "Prague": (50.0755, 14.4378),
    "Stockholm": (59.3293, 18.0686),
    "Oslo": (59.9139, 10.7522),
    "Helsinki": (60.1699, 24.9384),
    "Athens": (37.9838, 23.7275),
    "Moscow": (55.7558, 37.6173),

    # North America
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "San Francisco": (37.7749, -122.4194),
    "Seattle": (47.6062, -122.3321),
    "Washington DC": (38.9072, -77.0369),
    "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207),
    "Mexico City": (19.4326, -99.1332),

    # South America
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Buenos Aires": (-34.6037, -58.3816),
    "Lima": (-12.0464, -77.0428),
    "Santiago": (-33.4489, -70.6693),
    "Bogota": (4.7110, -74.0721),

    # Africa
    "Cairo": (30.0444, 31.2357),
    "Nairobi": (-1.2921, 36.8219),
    "Lagos": (6.5244, 3.3792),
    "Johannesburg": (-26.2041, 28.0473),
    "Cape Town": (-33.9249, 18.4241),
    "Casablanca": (33.5731, -7.5898),

    # Australia and Oceania
    "Sydney": (-33.8688, 151.2093),
    "Melbourne": (-37.8136, 144.9631),
    "Perth": (-31.9505, 115.8605),
    "Brisbane": (-27.4698, 153.0251),
    "Auckland": (-36.8509, 174.7645),
    "Wellington": (-41.2865, 174.7762)
}
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

# Load flood mo
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
            "minmagnitude": 0,
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