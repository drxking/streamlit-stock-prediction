import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd

def fetch_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data = data.tz_localize(None)  # Remove timezone information
    return data


def plot_stock_data(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name="Stock Open"))
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Stock Close"))
    
    fig.update_layout(
        title=f"Stock Price Chart for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
    )
    
    st.plotly_chart(fig)

def predict_stock(data, ticker, periods=30):
    df = data.reset_index()[['Date', 'Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)  # Ensure no timezone in the date column
    df.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Predicted Close Price"))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name="Actual Close Price"))
    
    fig.update_layout(
        title=f"Stock Price Prediction for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
        
    )
    
    st.plotly_chart(fig)

def main():
    st.title("Stock Price Prediction")
    
    tickers = ['BTC-USD','GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA']
    selected_ticker = st.selectbox("Choose a stock:", tickers, format_func=lambda x: x)
    
    
    
    
    data = fetch_stock_data(selected_ticker, "4y")
    if not data.empty:
        st.write("### Raw Data")
        st.dataframe(data,height=210, use_container_width=True)  
        plot_stock_data(data, selected_ticker)
        
        st.write("### Stock Price Prediction")
        predict_stock(data, selected_ticker)
    else:
        st.error("Failed to fetch data. Please try again later.")

if __name__ == "__main__":
    main()