import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import snowflake.connector
import os

st.set_page_config(page_title="📈 Forecast for Next 32 Months", layout="centered")
st.title("📈 Financial Forecasting App")
st.markdown("Using Prophet to forecast the next 32 months based on Snowflake data")

# Snowflake connection using Streamlit secrets
conn = snowflake.connector.connect(
    user=st.secrets["user"],
    password=st.secrets["password"],
    account=st.secrets["account"],
    warehouse=st.secrets["warehouse"],
    database="database",
    schema="schema"
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



