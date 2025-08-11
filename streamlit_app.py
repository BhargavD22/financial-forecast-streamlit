import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import snowflake.connector
import os

st.set_page_config(page_title="📈 Forecast for Next 32 Months", layout="centered")
st.title("📈 Financial Forecasting App")
st.markdown("Using Prophet to forecast the next 32 months based on Snowflake data")
# 🔍 Debug: Check what secrets are available
st.write("Secrets keys available:", list(st.secrets.keys()))
if "snowflake" in st.secrets:
    st.write("Snowflake keys:", list(st.secrets["snowflake"].keys()))
else:
    st.write("No 'snowflake' section found in secrets!")

# Snowflake connection using Streamlit secrets
conn = snowflake.connector.connect(
    user=st.secrets["snowflake"]["user"],
    password=st.secrets["snowflake"]["password"],
    account=st.secrets["snowflake"]["account"],
    warehouse=st.secrets["snowflake"]["warehouse"],
    database=st.secrets["snowflake"]["database"],
    schema=st.secrets["snowflake"]["schema"],
)


# Fetch data from Snowflake
query = "SELECT ds, y FROM forecast_data ORDER BY ds"
df = pd.read_sql(query, conn)
conn.close()

# Show raw data
st.subheader("Raw Data")
st.write(df.tail())

# Forecast using Prophet
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=32, freq='M')
forecast = model.predict(future)

# Plot
st.subheader("Forecast Chart")
fig = model.plot(forecast)
st.pyplot(fig)






