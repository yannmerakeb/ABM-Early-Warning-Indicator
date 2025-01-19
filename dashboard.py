import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessor import *
from tools import *

def main():
    # Sidebar configuration
    st.sidebar.title("Configuration")

    # New: Index selection input
    st.sidebar.subheader("Choose Index")
    index_name = st.sidebar.text_input("Index Name", value=["S&P500", "Nikkei225", "Eurostoxx600", "FTSE100", "TSX", "ASX200", "JSE"][0])

    selected_index = st.sidebar.selectbox(f"Select {index_name} or one of the stock composing it", [f"{index_name}"] + ["Stock"])

    # Main title
    st.title("Financial Analysis Dashboard")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Prices & Returns", "Parametric Estimations", "Distribution Comparison", "Settings"])

    # Load data based on user-selected index
    data = load_data(index_name)  # Updated function

    # Prices & Returns Tab
    with tab1:
        st.header("Prices and Returns")

        is_index = selected_index == "Index"

        returns_and_prices = ReturnsAndPrices(data)
        prices = returns_and_prices.index if is_index else returns_and_prices.stocks[selected_index]
        cumulative_returns = returns_and_prices.cumulative_returns(is_index)
        returns = returns_and_prices.returns(is_index)

        st.subheader("Prices")
        st.line_chart(prices)

        st.subheader("Cumulative Returns")
        st.line_chart(cumulative_returns)

        st.subheader("Daily Returns")
        st.line_chart(returns)

    # Parametric Estimations Tab
    with tab2:
        st.header("Parametric Estimations")

        likelihood = Likelihood(data)

        distribution = st.selectbox("Choose Distribution", ["Beta", "Normal"])
        drop_threshold = st.slider("Drop Threshold", 0.0, 1.0, 0.95, 0.01)

        params = likelihood.MLE(distribution.lower(), drop_threshold)

        st.subheader(f"{distribution} Parameters")
        st.write(params)

    # Distribution Comparison Tab
    with tab3:
        st.header("Empirical vs Theoretical Distribution")

        likelihood = Likelihood(data)

        distribution = st.selectbox("Choose Distribution", ["Beta", "Normal"])
        drop_threshold = st.slider("Drop Threshold", 0.0, 1.0, 0.95, 0.01)

        likelihood.theoretical_vs_empirical(distribution.lower(), drop_threshold)

        st.pyplot()

    # Settings Tab
    with tab4:
        st.header("Settings")

        ema_window = st.slider("EMA Window", 10, 200, 100, 10)
        sentiment_index = SentimentIndex(data, EMA_window=ema_window)

        st.subheader("Sentiment Index")
        st.line_chart(sentiment_index.sentiment_index["Sentiment Index"])

        st.subheader("Autocorrelation")
        sentiment_index.autocorrelation()
        st.pyplot()


def load_data(index_name):
    """
    Load and preprocess data based on user-selected index name.
    """
    # Initialize the DataPreprocessor with the given index_name
    dpp = DataPreprocessor('data', index_name)
    return dpp.process_csv_files()

if __name__ == "__main__":
    main()
