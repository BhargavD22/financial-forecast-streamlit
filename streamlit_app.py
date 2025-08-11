import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import snowflake.connector

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(page_title="📈 Forecast for Next 32 Months", layout="centered")
st.title("📈 Financial Forecasting App")
st.markdown("Using Prophet to forecast the next 32 months based on Snowflake data")

# -------------------
# Snowflake Connection
# -------------------
try:
    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
    )
    st.success("✅ Connected to Snowflake successfully!")
except Exception as e:
    st.error(f"❌ Failed to connect to Snowflake: {e}")
    st.stop()

# -------------------
# Fetch Data from Snowflake
# -------------------
query = "SELECT ds, y FROM forecast_data ORDER BY ds"
df = pd.read_sql(query, conn)
conn.close()

# -------------------
# Ensure Correct Columns
# -------------------
df.columns = df.columns.str.lower().str.strip()  # Normalize column names
if not set(['ds', 'y']).issubset(df.columns):
    st.error("❌ Dataframe must have columns 'ds' and 'y'. Found: " + str(df.columns.tolist()))
    st.stop()

# -------------------
# Type Conversion
# -------------------
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Drop rows with missing values
df = df.dropna(subset=['ds', 'y'])

if df.empty:
    st.error("❌ No valid data after cleaning. Check your Snowflake table.")
    st.stop()

## Show raw data
# st.subheader("Raw Data Preview")
# st.write(df.tail())
# -------------------
# User selects forecast horizon
# -------------------
forecast_days = st.slider("Select Forecast Horizon (Days)", min_value=30, max_value=365, value=90, step=1)

# -------------------
# Prophet Forecast
# -------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=forecast_days, freq='D')
forecast = model.predict(future)

# -------------------
# Plot
# -------------------
st.subheader(f"Forecast for Next {forecast_days} Days")
fig = model.plot(forecast)
st.pyplot(fig)


## -------------------
# Forecasting
# -------------------
try:
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=32, freq='M')
    forecast = model.predict(future)

    # Plot
    st.subheader("Forecast Chart")
    fig = model.plot(forecast)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Forecasting failed: {e}")


