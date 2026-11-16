import os
import math
from io import StringIO
from datetime import datetime, timedelta

import joblib
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Multi-Hazard Early Warning System",
    page_icon="🌍",
    layout="wide"
)
# =========================================================
# GLOBAL HISTORICAL FLOOD ANALYSIS
# =========================================================

st.header("🌍 Global Historical Flood Analysis")

historical_file = os.path.join(
    "historical_data",
    "Global_Flood_Records.csv"
)

if os.path.exists(historical_file):

    try:
        historical_df = pd.read_csv(historical_file)

        st.success("Historical Flood Dataset Loaded Successfully!")

        # Clean column names
        historical_df.columns = (
            historical_df.columns.str.strip()
        )

        # Convert numeric columns
        for col in [
            "Fatalities",
            "Displaced",
            "Severity",
            "Flood impact index"
        ]:
            historical_df[col] = pd.to_numeric(
                historical_df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )

        # Summary metrics
        total_events = len(historical_df)
        total_fatalities = historical_df["Fatalities"].sum()
        total_displaced = historical_df["Displaced"].sum()
        average_severity = historical_df["Severity"].mean()
        average_impact = historical_df["Flood impact index"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Flood Events", f"{total_events:,}")
        c2.metric("Total Fatalities", f"{total_fatalities:,.0f}")
        c3.metric("Total Displaced", f"{total_displaced:,.0f}")

        c4, c5 = st.columns(2)

        c4.metric(
            "Average Severity",
            f"{average_severity:.2f}" if pd.notna(average_severity) else "N/A"
        )

        c5.metric(
            "Average Flood Impact Index",
            f"{average_impact:.2f}" if pd.notna(average_impact) else "N/A"
        )
  
        # =========================================================
        # COUNTRY-WISE FLOOD ANALYSIS
        # =========================================================

        st.subheader("🌍 Country-wise Flood Analysis")

        countries = sorted(
            historical_df["Country"].dropna().astype(str).unique()
        )

        selected_country = st.selectbox(
            "Select Country",
            ["All Countries"] + countries
        )

        # Filter records
        if selected_country == "All Countries":
            filtered_df = historical_df.copy()
        else:
            filtered_df = historical_df[
                historical_df["Country"].astype(str) == selected_country
            ].copy()

        # Country-wise metrics
        st.subheader(f"Flood Statistics: {selected_country}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Flood Events", f"{len(filtered_df):,}")

        col2.metric(
            "Fatalities",
            f"{filtered_df['Fatalities'].sum():,.0f}"
        )

        col3.metric(
            "Displaced People",
            f"{filtered_df['Displaced'].sum():,.0f}"
        )

        col4, col5 = st.columns(2)

        avg_severity = filtered_df["Severity"].mean()
        avg_impact = filtered_df["Flood impact index"].mean()

        col4.metric(
            "Average Severity",
            f"{avg_severity:.2f}" if pd.notna(avg_severity) else "N/A"
        )

        col5.metric(
            "Average Impact Index",
            f"{avg_impact:.2f}" if pd.notna(avg_impact) else "N/A"
        )
        # COUNTRY-WISE FLOOD BAR GRAPH
        st.subheader("📊 Flood Events by Country")

        country_counts = (
            filtered_df["Country"]
            .value_counts()
            .head(15)
        )

        st.bar_chart(country_counts)

        # HISTORICAL FLOOD SEVERITY GRAPH
        st.subheader("📊 Historical Flood Severity")

        severity_counts = (
            filtered_df["Severity"]
            .dropna()
            .round(1)
            .value_counts()
            .sort_index()
        )

        st.bar_chart(severity_counts)

        # HISTORICAL FLOOD RECORDS
        st.subheader("📋 Historical Flood Records")

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True
     
        )
    # =========================================================
        # YEAR-WISE HISTORICAL FLOOD ANALYSIS
        # =========================================================

        st.subheader("📅 Year-wise Historical Flood Analysis")

        yearly_df = filtered_df.copy()

        yearly_df["Start Date"] = pd.to_datetime(
            yearly_df["Start Date"],
            errors="coerce"
        )

        yearly_df["Year"] = yearly_df["Start Date"].dt.year

        available_years = sorted(
            yearly_df["Year"].dropna().unique().astype(int),
            reverse=True
        )

        selected_year = st.selectbox(
            "Select Year",
            ["All Years"] + available_years
        )

        if selected_year == "All Years":
            year_filtered_df = yearly_df.copy()
        else:
            year_filtered_df = yearly_df[
                yearly_df["Year"] == selected_year
            ]

        yearly_counts = (
            year_filtered_df["Year"]
            .value_counts()
            .sort_index()
        )

        st.write("Flood Events by Year")
        st.bar_chart(yearly_counts)

        st.metric(
            "Flood Events in Selected Year",
            f"{len(year_filtered_df):,}"
        )





    except Exception as error:
        st.error(f"Historical Flood Analysis Error: {error}")

else:
    st.warning(
        "Historical flood dataset not found. "
        "Please check the historical_data folder."
    )

st.title("🌍 AI Multi-Hazard Early Warning System")
st.write("AI-based multi-hazard monitoring and risk-analysis prototype")

st.info(
    "This application is an educational prototype, not an official warning "
    "service. Always follow government alerts and local emergency authorities."
)

st.subheader("🛡️ Official Disaster Information")
st.markdown("[🇮🇳 NDMA SACHET Official Alert Portal](https://sachet.ndma.gov.in/)")
st.markdown(
    "[🌊 CWC River Water-Level Dataset - National Water Data Portal]"
    "(https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc)"
)

# =========================================================
# CITY COORDINATES
# =========================================================
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
    "Agra": (27.1767, 78.0080),
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


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two coordinates in km."""
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))

  # =========================================================
# REAL GLOBAL FLOOD MONITORING - GloFAS + OPEN-METEO
# =========================================================

st.header("🌊 Real Global Flood Monitoring")

st.write(
    "Retrieve global river discharge observations and "
    "rainfall forecast for a selected location."
)

flood_location = st.text_input(
    "Enter location for flood monitoring",
    placeholder="Example: Greater Noida, India",
    key="global_flood_search_location"
)

if st.button("Fetch Real Flood Monitoring Data"):

    if not flood_location.strip():
        st.warning("Please enter a location.")

    else:
        try:
            # STEP 1: Find location coordinates
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": flood_location.strip(),
                    "count": 1,
                    "language": "en",
                    "format": "json"
                },
                timeout=20
            )

            geo_response.raise_for_status()
            geo_data = geo_response.json()

            results = geo_data.get("results", [])

            if not results:
                st.error("Location not found. Try another city.")

            else:
                place = results[0]
                latitude = place["latitude"]
                longitude = place["longitude"]

                st.success(
                    f"Location found: {place['name']}, "
                    f"{place.get('country', '')}"
                )

                # STEP 2: Fetch global river discharge data
                river_response = requests.get(
                    "https://flood-api.open-meteo.com/v1/flood",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "daily": "river_discharge",
                        "past_days": 30,
                        "forecast_days": 30,
                        "timezone": "GMT"
                    },
                    timeout=30
                )

                river_response.raise_for_status()
                river_data = river_response.json()

                river_daily = river_data.get("daily", {})
                river_dates = river_daily.get("time", [])
                river_values = river_daily.get(
                    "river_discharge", []
                )

                if river_dates and river_values:
                    river_df = pd.DataFrame({
                        "Date": river_dates,
                        "River Discharge (m³/s)": river_values
                    })

                    river_df["River Discharge (m³/s)"] = (
                        pd.to_numeric(
                            river_df["River Discharge (m³/s)"],
                            errors="coerce"
                        )
                    )

                    st.subheader("🌊 River Discharge Data")
                    st.dataframe(river_df, use_container_width=True)

                    st.line_chart(
                        river_df.set_index("Date")[
                            ["River Discharge (m³/s)"]
                        ]
                    )

                    st.session_state["global_river_data"] = river_df

                else:
                    st.warning(
                        "River discharge data is not available "
                        "for this location."
                    )

                # STEP 3: Fetch rainfall forecast
                rain_response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": latitude,
                        "longitude": longitude,
                        "daily": (
                            "precipitation_sum,"
                            "precipitation_probability_max"
                        ),
                        "forecast_days": 7,
                        "timezone": "auto"
                    },
                    timeout=30
                )

                rain_response.raise_for_status()
                rain_data = rain_response.json()

                rain_daily = rain_data.get("daily", {})

                if rain_daily.get("time"):
                    rain_df = pd.DataFrame({
                        "Date": rain_daily["time"],
                        "Rainfall (mm)": rain_daily.get(
                            "precipitation_sum", []
                        ),
                        "Rain Probability (%)": rain_daily.get(
                            "precipitation_probability_max", []
                        )
                    })

                    st.subheader("🌧️ Seven-Day Rainfall Forecast")
                    st.dataframe(
                        rain_df,
                        use_container_width=True
                    )

                    st.session_state["global_flood_rainfall"] = rain_df

                st.caption(
                    "Data sources: Open-Meteo Flood API (GloFAS) "
                    "and Open-Meteo Weather API."
                )

                st.info(
                    "This module displays monitoring and forecast "
                    "data. It does not generate an official flood warning."
                )
                # =====================================================
                # COMBINED GLOBAL FLOOD ANALYSIS DASHBOARD
                # =====================================================

                st.divider()
                st.header("🌊 Combined Flood Risk Analysis Dashboard")

                if (
                    "global_river_data" in st.session_state
                    and "global_flood_rainfall" in st.session_state
                ):

                    river_check = st.session_state[
                        "global_river_data"
                    ].copy()

                    rain_check = st.session_state[
                        "global_flood_rainfall"
                    ].copy()

                    # Convert data into numeric format
                    river_check["Date"] = pd.to_datetime(
                        river_check["Date"], errors="coerce"
                    )

                    river_check["River Discharge (m³/s)"] = pd.to_numeric(
                        river_check["River Discharge (m³/s)"],
                        errors="coerce"
                    )

                    rain_check["Rainfall (mm)"] = pd.to_numeric(
                        rain_check["Rainfall (mm)"],
                        errors="coerce"
                    )

                    rain_check["Rain Probability (%)"] = pd.to_numeric(
                        rain_check["Rain Probability (%)"],
                        errors="coerce"
                    )

                    river_check = river_check.dropna(
                        subset=["Date", "River Discharge (m³/s)"]
                    ).sort_values("Date")

                    # Use available river data up to today's date
                    today = pd.Timestamp.now().normalize()

                    observed_river = river_check[
                        river_check["Date"] <= today
                    ]

                    if not observed_river.empty:

                        latest_discharge = observed_river[
                            "River Discharge (m³/s)"
                        ].iloc[-1]

                        recent_values = observed_river[
                            "River Discharge (m³/s)"
                        ].tail(7)

                        if len(recent_values) >= 2:
                            if recent_values.iloc[-1] > recent_values.iloc[0]:
                                river_trend = "Increasing"
                            elif recent_values.iloc[-1] < recent_values.iloc[0]:
                                river_trend = "Decreasing"
                            else:
                                river_trend = "Stable"
                        else:
                            river_trend = "Insufficient data"

                    else:
                        latest_discharge = None
                        river_trend = "Insufficient historical data"

                    total_rainfall = rain_check[
                        "Rainfall (mm)"
                    ].sum()

                    maximum_probability = rain_check[
                        "Rain Probability (%)"
                    ].max()

                    # Display dashboard metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if latest_discharge is not None:
                            st.metric(
                                "Latest Available River Discharge",
                                f"{latest_discharge:.2f} m³/s"
                            )
                        else:
                            st.metric(
                                "River Discharge",
                                "Not available"
                            )

                    with col2:
                        st.metric(
                            "7-Day Forecast Rainfall",
                            f"{total_rainfall:.1f} mm"
                        )

                    with col3:
                        if pd.notna(maximum_probability):
                            st.metric(
                                "Maximum Rain Probability",
                                f"{maximum_probability:.0f}%"
                            )
                        else:
                            st.metric(
                                "Rain Probability",
                                "Not available"
                            )

                    # River trend
                    st.subheader("📈 River Discharge Trend")

                    if not observed_river.empty:
                        st.write(
                            f"Recent river discharge trend: *{river_trend}*"
                        )

                        st.line_chart(
                            observed_river.tail(14).set_index("Date")[
                                ["River Discharge (m³/s)"]
                            ]
                        )

                    # Rainfall trend
                    st.subheader("🌧️ Seven-Day Rainfall Analysis")

                    st.line_chart(
                        rain_check.set_index("Date")[
                            ["Rainfall (mm)"]
                        ]
                    )

                    st.dataframe(
                        rain_check,
                        use_container_width=True
                    )

                    st.subheader("⚠️ Preliminary Flood Indicators")

                    st.write(
                        f"- River discharge trend: {river_trend}"
                    )

                    st.write(
                        f"- Total forecast rainfall: "
                        f"{total_rainfall:.1f} mm"
                    )

                    st.info(
                        "These are preliminary monitoring indicators, "
                        "not validated flood-risk predictions. "
                        "Official flood thresholds and local river "
                        "conditions are required for reliable risk levels."
                    )

                else:
                    st.warning(
                        "Please fetch Global Flood Monitoring data first."
                    )

        except requests.exceptions.RequestException as error:
            st.error(f"Data connection error: {error}")

        except Exception as error:
            st.error(f"Something went wrong: {error}")
# =========================================================
# WORLDWIDE LIVE WEATHER
# =========================================================
st.header("🌍 Worldwide Live Weather Monitoring")

weather_location = st.text_input(
    "Enter any city or country",
    placeholder="Example: Tokyo, Japan"
)

if st.button("Fetch Worldwide Live Weather"):
    if not weather_location.strip():
        st.warning("Please enter a location.")
    else:
        try:
            geo_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={
                    "name": weather_location.strip(),
                    "count": 10,
                    "language": "en",
                    "format": "json"
                },
                timeout=20
            )
            geo_response.raise_for_status()
            locations = geo_response.json().get("results", [])

            if not locations:
                st.warning("Location not found. Try a city name.")
            else:
                selected = locations[0]
                latitude = selected["latitude"]
                longitude = selected["longitude"]

                weather_response = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
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
                    },
                    timeout=20
                )
                weather_response.raise_for_status()
                weather_data = weather_response.json()

                st.session_state["global_weather"] = {
                    "city": selected["name"],
                    "country": selected.get("country", "Unknown"),
                    "latitude": latitude,
                    "longitude": longitude,
                    "data": weather_data
                }
                st.success("Worldwide weather fetched successfully!")

        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Weather API error: {error}")

if "global_weather" in st.session_state:
    saved = st.session_state["global_weather"]
    current = saved["data"].get("current", {})
    daily = saved["data"].get("daily", {})

    st.subheader(f"Weather: {saved['city']}, {saved['country']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{current.get('temperature_2m', 'N/A')} °C")
    c2.metric("Humidity", f"{current.get('relative_humidity_2m', 'N/A')}%")
    c3.metric("Current precipitation", f"{current.get('precipitation', 'N/A')} mm")
    c4.metric("Wind speed", f"{current.get('wind_speed_10m', 'N/A')} km/h")

    st.write("Latitude:", saved["latitude"])
    st.write("Longitude:", saved["longitude"])

    if daily.get("time"):
        forecast_df = pd.DataFrame({
            "Date": daily.get("time", []),
            "Rainfall (mm)": daily.get("precipitation_sum", []),
            "Rain probability (%)": daily.get(
                "precipitation_probability_max", []
            )
        })
        st.subheader("3-Day Weather Forecast")
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)

        if st.button("Use Weather Data for Flood Analysis"):
            st.session_state["rainfall_input"] = float(
                daily["precipitation_sum"][0] or 0
            )
            st.session_state["temperature_input"] = float(
                current.get("temperature_2m", 0)
            )
            st.session_state["humidity_input"] = int(
                current.get("relative_humidity_2m", 0)
            )
            st.session_state["weather_input_city"] = saved["city"]
            st.success(
                f"Weather values transferred for {saved['city']}. "
                "Review the flood inputs below."
            )

    st.caption(
        "Source: Open-Meteo. Forecast values are model-based and are not "
        "official disaster warnings."
    )
    # =========================================================
# HISTORICAL RAINFALL
# =========================================================
st.header("📊 Historical Rainfall Analysis")

historical_city = st.selectbox(
    "Select city for historical rainfall",
    list(cities.keys()),
    key="historical_city"
)

if st.button("Fetch Last 7 Days Rainfall"):
    latitude, longitude = cities[historical_city]
    try:
        historical_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "precipitation_sum",
                "past_days": 7,
                "forecast_days": 1,
                "timezone": "auto"
            },
            timeout=20
        )
        historical_response.raise_for_status()
        daily_data = historical_response.json()["daily"]

        rainfall_df = pd.DataFrame({
            "Date": daily_data["time"],
            "Rainfall (mm)": daily_data["precipitation_sum"]
        })

        st.subheader(f"Rainfall History: {historical_city}")
        st.dataframe(rainfall_df, hide_index=True, use_container_width=True)
        st.line_chart(rainfall_df.set_index("Date")["Rainfall (mm)"])
        st.caption(
            "Source: Open-Meteo. Historical weather estimates are not "
            "official flood warnings."
        )
    except (requests.RequestException, KeyError, ValueError) as error:
        st.error(f"Historical rainfall fetch failed: {error}")


# =========================================================
# LOAD FLOOD MODEL SAFELY
# =========================================================
model = None
model_load_error = None

if os.path.exists("flood_model.pkl"):
    try:
        model = joblib.load("flood_model.pkl")
    except Exception as error:
        model_load_error = str(error)
else:
    model_load_error = "flood_model.pkl was not found in the project folder."

if model_load_error:
    st.warning(f"Flood model status: {model_load_error}")
    # =========================================================
# FLOOD MODEL PERFORMANCE - CROSS VALIDATION
# =========================================================
st.subheader("📈 Flood Model Performance")

if model is not None and os.path.exists("flood_data.csv"):
    try:
        data = pd.read_csv("flood_data.csv")

        data.columns = data.columns.str.strip()

        data["risk"] = (
            data["risk"]
            .astype(str)
            .str.strip()
            .str.capitalize()
        )

        features = ["rainfall", "temperature", "humidity"]

        for col in features:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

        data = data.dropna(
            subset=features + ["risk"]
        )

        X = data[features]
        y = data["risk"]

        if y.value_counts().min() < 5:
            st.warning(
                "Not enough samples in every risk class "
                "for 5-fold cross-validation."
            )
        else:
            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="accuracy"
            )

            average_accuracy = scores.mean() * 100

            st.metric(
                "Average 5-Fold Cross Validation Accuracy",
                f"{average_accuracy:.2f}%"
            )

           

            st.caption(
                "Accuracy is calculated using 5-fold cross-validation. "
                "This is an evaluation estimate, not a guarantee "
                "of real-world prediction accuracy."
            )

    except Exception as error:
        st.error(
            f"Flood model evaluation failed: {error}"
        )

elif not os.path.exists("flood_data.csv"):
    st.warning(
        "flood_data.csv not found."
    )



# =========================================================
# ANALYSIS HISTORY
# =========================================================
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []


# =========================================================
# HAZARD SELECTION AND INPUTS
# =========================================================
st.header("⚠️ Hazard Risk Analysis")

hazard = st.selectbox(
    "Select Hazard",
    ["Flood", "Heatwave", "Earthquake", "Landslide", "Wildfire","Cyclone"]
)

selected_hazard_city = None

if hazard in ["Earthquake", "Wildfire"]:
    selected_hazard_city = st.selectbox(
        f"Select city for {hazard} monitoring",
        list(cities.keys()),
        key=f"{hazard.lower()}_monitoring_city"
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
    "Earthquake minimum magnitude (USGS historical-data filter)",
    min_value=0.0,
    max_value=9.9,
    value=2.0,
    step=0.1
)

slope = st.slider(
    "Slope steepness (degrees)",
    min_value=0,
    max_value=90,
    value=15
)


# =========================================================
# ANALYZE HAZARD
# =========================================================
if st.button("Analyze Hazard Risk"):
    risk = None
    risk_detail = ""

    # ---------------- FLOOD ----------------
    if hazard == "Flood":
        if model is None:
            st.error(
                "Flood model is unavailable. Check flood_model.pkl "
                "and the error shown above."
            )
        else:
            try:
                input_data = pd.DataFrame(
                    [[rainfall, temperature, humidity]],
                    columns=["rainfall", "temperature", "humidity"]
                )

                expected_features = getattr(
                    model, "feature_names_in_", None
                )

                if expected_features is not None:
                    expected_features = list(expected_features)

                    if set(expected_features) != set(input_data.columns):
                        st.error(
                            "The saved model expects different input features: "
                            + ", ".join(expected_features)
                        )
                    else:
                        input_data = input_data[expected_features]
                        risk = str(model.predict(input_data)[0])
                else:
                    risk = str(model.predict(input_data)[0])

            except Exception as error:
                st.error(f"Flood prediction failed: {error}")

    # ---------------- HEATWAVE ----------------
    elif hazard == "Heatwave":
        if temperature >= 40:
            risk = "High"
        elif temperature >= 35:
            risk = "Moderate"
        else:
            risk = "Low"

        risk_detail = (
            "Illustrative temperature thresholds only; local heat-health "
            "warnings also depend on humidity, duration and official criteria."
        )
    # ---------------- CYCLONE ----------------
    elif hazard == "Cyclone":

        st.subheader("🌀 Worldwide Cyclone Formation Probability")

        st.info(
            "Live cyclone formation guidance from NOAA "
            "for the global tropical region."
        )

        st.markdown(
            "[🌍 Open NOAA Global Cyclone Forecast]"
            "(https://www.ospo.noaa.gov/products/ocean/tropical/tcfp/)"
        )

        st.components.v1.iframe(
            "https://www.ospo.noaa.gov/products/ocean/tropical/tcfp/",
            height=850,
            scrolling=True
        )

        risk = "Monitoring"

        risk_detail = (
            "NOAA provides 24-hour and 48-hour cyclone "
            "formation probability guidance. This is forecast "
            "guidance, not a guaranteed cyclone prediction."
        )

    # ---------------- EARTHQUAKE ----------------
    elif hazard == "Earthquake":
        latitude, longitude = cities[selected_hazard_city]

        analysis_radius_km = 300
        history_days = 365

        history_start = (
            datetime.utcnow() - timedelta(days=history_days)
        ).strftime("%Y-%m-%d")

        params = {
            "format": "geojson",
            "latitude": latitude,
            "longitude": longitude,
            "maxradiuskm": analysis_radius_km,
            "starttime": history_start,
            "minmagnitude": float(magnitude),
            "orderby": "time",
            "limit": 20000
        }

        try:
            with st.spinner(
                "Fetching one year of historical earthquake records from USGS..."
            ):
                response = requests.get(
                    "https://earthquake.usgs.gov/fdsnws/event/1/query",
                    params=params,
                    timeout=45
                )
                response.raise_for_status()
                events = response.json().get("features", [])

            records = []

            for event in events:
                properties = event.get("properties") or {}
                geometry = event.get("geometry") or {}
                coordinates = geometry.get("coordinates") or []

                if properties.get("mag") is None or len(coordinates) < 2:
                    continue

                event_lon = float(coordinates[0])
                event_lat = float(coordinates[1])

                event_depth = (
                    float(coordinates[2])
                    if len(coordinates) >= 3
                    and coordinates[2] is not None
                    else None
                )

                distance = haversine_km(
                    latitude, longitude, event_lat, event_lon
                )

                event_time_ms = properties.get("time")

                event_datetime = (
                    datetime.utcfromtimestamp(event_time_ms / 1000)
                    if event_time_ms else None
                )

                records.append({
                    "Magnitude": float(properties["mag"]),
                    "Location": properties.get("place", "Unknown"),
                    "Distance from city (km)": round(distance, 2),
                    "Depth (km)": event_depth,
                    "Time (UTC)": (
                        event_datetime.strftime("%Y-%m-%d %H:%M")
                        if event_datetime else "Unknown"
                    ),
                    "Event datetime": event_datetime,
                    "USGS Details": properties.get("url", "")
                })

            earthquake_df = pd.DataFrame(records)

            if earthquake_df.empty:
                st.info(
                    f"No events of magnitude {magnitude:.1f}+ were returned "
                    f"within {analysis_radius_km} km of {selected_hazard_city} "
                    f"during the last {history_days} days."
                )

                risk = "Not available"
                risk_detail = (
                    "Insufficient matching historical events for this estimate. "
                    "This does not mean future earthquake risk is zero."
                )

            else:
                earthquake_df = earthquake_df.sort_values(
                    "Distance from city (km)"
                )

                st.subheader(
                    f"Historical Earthquake Records near {selected_hazard_city}"
                )

                st.write(
                    f"Records: {len(earthquake_df)} | Search radius: "
                    f"{analysis_radius_km} km | Historical window: "
                    f"{history_days} days | Minimum magnitude: {magnitude:.1f}"
                )

                display_df = earthquake_df.drop(
                    columns=["Event datetime"], errors="ignore"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Historical event rate and Poisson baseline estimates.
                event_count_365 = len(earthquake_df)
                daily_rate = event_count_365 / float(history_days)

                probability_7 = (
                    1 - math.exp(-daily_rate * 7)
                ) * 100

                probability_30 = (
                    1 - math.exp(-daily_rate * 30)
                ) * 100

                st.subheader(
                    "🧪 Experimental 7-Day and 30-Day Seismic Activity Estimates"
                )

                p1, p2, p3 = st.columns(3)
                p1.metric("Historical matching events", event_count_365)
                p2.metric(
                    "7-day baseline probability",
                    f"{probability_7:.2f}%"
                )
                p3.metric(
                    "30-day baseline probability",
                    f"{probability_30:.2f}%"
                )

                def estimate_band(probability):
                    if probability < 5:
                        return "Low"
                    elif probability < 20:
                        return "Moderate"
                    return "Elevated"

                band_7 = estimate_band(probability_7)
                band_30 = estimate_band(probability_30)

                st.write("*7-day experimental activity band:*", band_7)
                st.write("*30-day experimental activity band:*", band_30)

                # Compare recent 30-day activity with previous 335 days.
                now_utc = datetime.utcnow()
                event_dates = earthquake_df["Event datetime"].dropna()

                recent_30_count = int(
                    (event_dates >= now_utc - timedelta(days=30)).sum()
                )

                previous_335_count = int(
                    (
                        (event_dates < now_utc - timedelta(days=30))
                        & (event_dates >= now_utc - timedelta(days=365))
                    ).sum()
                )

                previous_30_day_equivalent = (
                    previous_335_count / (335 / 30)
                )

                st.write("*Recent activity comparison*")
                st.write(f"Events in latest 30 days: {recent_30_count}")
                st.write(
                    "Average 30-day-equivalent events in preceding 335 days: "
                    f"{previous_30_day_equivalent:.2f}"
                )

                nearest_distance = float(
                    earthquake_df["Distance from city (km)"].min()
                )

                strongest_magnitude = float(
                    earthquake_df["Magnitude"].max()
                )

                st.write(
                    f"Nearest recorded event: {nearest_distance:.2f} km away"
                )

                st.write(
                    f"Strongest recorded event in the search area: "
                    f"M{strongest_magnitude:.1f}"
                )

                risk = band_30

                risk_detail = (
                    "Experimental historical-rate estimate only. It assumes "
                    "the past year's event rate remains constant and events "
                    "follow a simplified Poisson process. It does NOT predict "
                    "whether, when, where, or how strongly an earthquake will occur."
                )

                st.warning(
                    "Scientific limitation: earthquakes cannot currently be "
                    "reliably predicted for an exact city and future 7- or 30-day "
                    "window. These percentages are historical-rate illustrations, "
                    "not validated forecasts or probabilities of a damaging earthquake."
                )

        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"USGS earthquake data fetch failed: {error}")
            risk = None
            # ---------------- LANDSLIDE ----------------
    elif hazard == "Landslide":
        if rainfall >= 100 and slope >= 30:
            risk = "High"
        elif rainfall >= 50 and slope >= 20:
            risk = "Moderate"
        else:
            risk = "Low"

        risk_detail = (
            "Illustrative rainfall and slope thresholds only. Actual "
            "landslide risk depends on soil, geology, drainage and local conditions."
        )

    # ---------------- WILDFIRE ----------------
    elif hazard == "Wildfire":
        if "wildfire_data" not in st.session_state:
            st.warning(
                "Fetch NASA wildfire data first using the section below."
            )
        else:
            fire_df = st.session_state["wildfire_data"].copy()
            required_fire_columns = {"latitude", "longitude", "frp"}

            if not required_fire_columns.issubset(fire_df.columns):
                st.warning(
                    "NASA data is missing one or more required columns: "
                    "latitude, longitude, frp."
                )
            else:
                for col in ["latitude", "longitude", "frp"]:
                    fire_df[col] = pd.to_numeric(
                        fire_df[col], errors="coerce"
                    )

                fire_df = fire_df.dropna(
                    subset=["latitude", "longitude", "frp"]
                )

                city_lat, city_lon = cities[selected_hazard_city]

                if fire_df.empty:
                    st.warning(
                        "No valid NASA fire hotspot records are available."
                    )
                else:
                    fire_df["Distance from city (km)"] = fire_df.apply(
                        lambda row: haversine_km(
                            city_lat,
                            city_lon,
                            row["latitude"],
                            row["longitude"]
                        ),
                        axis=1
                    )

                    local_fires = fire_df[
                        fire_df["Distance from city (km)"] <= 100
                    ].copy()

                    st.write(
                        f"Fire hotspots within 100 km of "
                        f"{selected_hazard_city}: {len(local_fires)}"
                    )

                    if local_fires.empty:
                        risk = "Low"
                        risk_detail = (
                            "No valid satellite hotspot was found within "
                            "100 km in the downloaded data. Satellite coverage "
                            "and detection limitations apply; this is not a "
                            "guarantee that no fire exists."
                        )
                    else:
                        max_frp = float(local_fires["frp"].max())

                        if max_frp >= 50:
                            risk = "High"
                        elif max_frp >= 10:
                            risk = "Moderate"
                        else:
                            risk = "Low"

                        risk_detail = (
                            f"{len(local_fires)} hotspot(s) within 100 km. "
                            f"Maximum FRP: {max_frp:.2f} MW. "
                            "This is a simple satellite-hotspot indicator, "
                            "not a validated wildfire forecast."
                        )

                        st.dataframe(
                            local_fires.sort_values(
                                "Distance from city (km)"
                            ).head(100),
                            use_container_width=True,
                            hide_index=True
                        )

    # ---------------- DISPLAY AND SAVE RESULT ----------------
    if risk is not None:
        st.subheader("Analysis Result")
        st.write("Selected hazard:", hazard)

        if risk == "High":
            st.error("🚨 Demo Risk Category: HIGH")
        elif risk == "Moderate":
            st.warning("⚠️ Demo Risk Category: MODERATE")
        elif risk == "Low":
            st.success("✅ Demo Risk Category: LOW")
        else:
            st.info(f"Result: {risk}")

        if risk_detail:
            st.caption(risk_detail)

        record = {
            "Date and Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hazard": hazard,
            "Monitoring City": selected_hazard_city or (
                st.session_state.get("weather_input_city", "Manual input")
            ),
            "Rainfall (mm)": rainfall,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "Minimum Earthquake Magnitude": magnitude,
            "Slope (degrees)": slope,
            "Demo Risk Result": risk,
            "Details": risk_detail
        }

        st.session_state["analysis_history"].append(record)

        st.caption(
            "Flood uses the saved ML model. Heatwave and landslide use "
            "illustrative rules. Earthquake uses recent USGS observations. "
            "Wildfire uses NASA FIRMS hotspot observations. None of these "
            "outputs is an official emergency forecast."
        )


# =========================================================
# INPUT SUMMARY
# =========================================================
st.subheader("Input Data Summary")

input_summary = pd.DataFrame({
    "Parameter": [
        "Rainfall (mm)",
        "Temperature (°C)",
        "Humidity (%)",
        "Earthquake minimum magnitude",
        "Slope (degrees)"
    ],
    "Value": [rainfall, temperature, humidity, magnitude, slope]
})

st.dataframe(
    input_summary,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# ANALYSIS HISTORY AND CSV DOWNLOAD
# =========================================================
st.subheader("📋 Analysis History")

if st.session_state["analysis_history"]:
    history_df = pd.DataFrame(st.session_state["analysis_history"])

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Analysis Report (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="hazard_analysis_report.csv",
        mime="text/csv"
    )

    if st.button("Clear Analysis History"):
        st.session_state["analysis_history"] = []
        st.rerun()

else:
    st.info(
        "No analyses yet. Click Analyze Hazard Risk to create your first record."
    )

# =========================================================
# WORLDWIDE WILDFIRE MONITORING - NASA FIRMS
# =========================================================
st.subheader("🔥 Worldwide Wildfire Monitoring")

st.write(
    "View satellite-detected fire hotspots around the world "
    "using NASA FIRMS data."
)

if st.button("Fetch Latest Wildfire Data"):
    try:
        nasa_key = st.secrets.get("NASA_FIRMS_MAP_KEY", "")

        if not nasa_key:
            st.error("NASA FIRMS API key is missing.")
        else:
            fire_df = pd.DataFrame()

            # Current satellite source with backup
            sources = [
                "VIIRS_NOAA21_NRT",
                "VIIRS_NOAA20_NRT"
            ]

            for source in sources:
                wildfire_url = (
                    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                    f"{nasa_key}/{source}/world/3"
                )

                with st.spinner(
                    f"Fetching global fire data from {source}..."
                ):
                    response = requests.get(
                        wildfire_url,
                        timeout=90
                    )

                st.write(
                    f"{source} HTTP Status:",
                    response.status_code
                )

                response.raise_for_status()

                if response.text.strip():
                    temp_df = pd.read_csv(
                        StringIO(response.text)
                    )

                    temp_df.columns = [
                        str(col).strip().lower()
                        for col in temp_df.columns
                    ]

                    if (
                        not temp_df.empty
                        and {"latitude", "longitude", "frp"}
                        .issubset(temp_df.columns)
                    ):
                        fire_df = temp_df
                        st.success(
                            f"Data received from {source}"
                        )
                        break

            if fire_df.empty:
                st.warning(
                    "No fire hotspot records were returned "
                    "by the available satellite sources."
                )
                st.session_state["wildfire_data"] = pd.DataFrame(
                    columns=["latitude", "longitude", "frp"]
                )
            else:
                for col in ["latitude", "longitude", "frp"]:
                    fire_df[col] = pd.to_numeric(
                        fire_df[col],
                        errors="coerce"
                    )

                fire_df = fire_df.dropna(
                    subset=["latitude", "longitude", "frp"]
                )

                st.session_state["wildfire_data"] = fire_df

                st.success(
                    "Worldwide wildfire data fetched successfully!"
                )

                st.write(
                    "Total fire hotspots:",
                    len(fire_df)
                )

    except Exception as error:
        st.error(f"Wildfire data error: {error}")


# =========================================================
# WILDFIRE STATISTICS AND GLOBAL MAP
# =========================================================
if "wildfire_data" in st.session_state:
    fire_df = st.session_state["wildfire_data"].copy()

    st.markdown("### 🔥 Wildfire Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Fire Hotspots",
        len(fire_df)
    )

    if "confidence" in fire_df.columns:
        confidence = (
            fire_df["confidence"]
            .astype(str)
            .str.upper()
        )

        high_confidence = confidence.isin(
            ["H", "HIGH"]
        ).sum()

        col2.metric(
            "High Confidence Fires",
            int(high_confidence)
        )
    else:
        col2.metric(
            "High Confidence Fires",
            "N/A"
        )

    if "frp" in fire_df.columns:
        frp_values = pd.to_numeric(
            fire_df["frp"],
            errors="coerce"
        ).dropna()

        average_frp = (
            frp_values.mean()
            if not frp_values.empty
            else None
        )

        col3.metric(
            "Average FRP",
            f"{average_frp:.2f} MW"
            if average_frp is not None
            else "N/A"
        )
    else:
        col3.metric(
            "Average FRP",
            "N/A"
        )

    st.markdown("### 🌍 Global Fire Hotspot Map")

    if {"latitude", "longitude"}.issubset(fire_df.columns):
        map_df = fire_df[
            ["latitude", "longitude"]
        ].copy()

        map_df["latitude"] = pd.to_numeric(
            map_df["latitude"],
            errors="coerce"
        )

        map_df["longitude"] = pd.to_numeric(
            map_df["longitude"],
            errors="coerce"
        )

        map_df = map_df.dropna()

        if not map_df.empty:
            st.map(map_df.head(5000))
        else:
            st.warning(
                "No valid map coordinates available."
            )

    st.markdown("### 📊 Fire Detection Records")

    st.dataframe(
        fire_df,
        use_container_width=True
    )

    st.download_button(
        "Download Wildfire Data CSV",
        data=fire_df.to_csv(index=False).encode("utf-8"),
        file_name="worldwide_wildfire_data.csv",
        mime="text/csv"
     
    )        
    

    st.caption(
        "NASA FIRMS satellite detections are hotspot observations, not "
        "confirmed wildfire incidents or official emergency warnings."
    )

   