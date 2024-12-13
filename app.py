import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.title("Stock Price Prediction with ARIMA")

# Sidebar for inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365), max_value=datetime.now())
end_date = st.sidebar.date_input("End Date", datetime.now())
future_steps = st.sidebar.slider("Days to Predict:", 1, 30, 7)

# Fetch data
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Close' in data.columns:
            data = data['Close']
        else:
            return pd.Series()  # Empty series if 'Close' column is missing
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.Series()  # Return empty series in case of any error

    data.index = pd.to_datetime(data.index)
    return data

stock_data = get_stock_data(ticker, start_date, end_date)

if stock_data.empty:
    st.error("Unable to fetch valid stock data. Please verify the stock ticker and date range.")
else:
    st.subheader(f"Closing Prices for {ticker}")
    st.line_chart(stock_data)

    # Train ARIMA model
    st.subheader("ARIMA Model Training")
    if len(stock_data) >= 30:  # Ensure we have enough data to fit the model
        try:
            # Fit the ARIMA model
            fitted_model = ARIMA(stock_data, order=(5, 2, 0))
            fitted_model = fitted_model.fit()
            st.write("ARIMA Model Summary:")
            st.write(fitted_model.summary())
        except Exception as e:
            st.error(f"Model training failed: {e}")

        # Forecast future stock prices
        st.subheader("Predicted Stock Prices")
        forecast_dates = pd.date_range(start=stock_data.index[-1], periods=future_steps + 1, freq='B')[1:]  # Ensure forecast dates are correctly aligned
        forecast = fitted_model.forecast(steps=future_steps)

        # Plot figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[ticker], mode='lines', name='Historical Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
    else:
        st.warning("Insufficient data to train the model. Please choose a longer date range or different stock.")