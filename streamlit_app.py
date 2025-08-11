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

# Fetch data from Snowflake
query = "SELECT ds, y FROM forecast_data ORDER BY ds"
df = pd.read_sql(query, conn)
conn.close()

# Show raw data
st.subheader("Raw Data")
st.write(df.tail())

## adjust these names to match your Snowflake table
df = df.rename(columns={
    "ds": "ds",    # replace with your actual date column name
    "y": "y"     # replace with your actual numeric column name
})

# Make sure types are correct
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'], errors='coerce')


# Forecast using Prophet
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=32, freq='M')
forecast = model.predict(future)

# Plot
st.subheader("Forecast Chart")
fig = model.plot(forecast)
st.pyplot(fig)










