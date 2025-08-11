import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import snowflake.connector
from io import BytesIO

# -------------------
# Streamlit Page Config
# -------------------
st.set_page_config(page_title="📈 Financial Forecasting App", layout="wide")
st.title("📈 Financial Forecasting App")
st.markdown("Forecast your data from Snowflake with Prophet")

# -------------------
# Connect to Snowflake
# -------------------
@st.cache_data
def load_data():
    try:
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            password=st.secrets["snowflake"]["password"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
        )
        query = "SELECT ds, y FROM forecast_data ORDER BY ds"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"❌ Failed to connect to Snowflake: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Ensure correct formats
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# -------------------
# User chooses forecast horizon
# -------------------
forecast_days = st.slider(
    "Select Forecast Horizon (Days)", 
    min_value=30, max_value=365, value=90, step=1
)

# -------------------
# Historical Trend Chart
# -------------------
st.subheader("📊 Historical Data Trend")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines+markers', name='Historical Data'))
fig_hist.update_layout(title="Historical Trend", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(fig_hist, use_container_width=True)

# -------------------
# Prophet Forecast
# -------------------
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=forecast_days, freq='D')
forecast = model.predict(future)

# Interactive Prophet forecast chart
st.subheader(f"🔮 Prophet Forecast (Next {forecast_days} Days)")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# -------------------
# Forecast Table
# -------------------
st.subheader("📄 Forecast Data Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# -------------------
# Download CSV Button
# -------------------
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv"
)
