# Standard python imports
from enum import Enum
from datetime import datetime, timedelta

# Third party imports
import streamlit as st
import yfinance as yf

# Local package imports
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

@st.cache_data
def get_historical_data(ticker):
    try:
        data = Ticker.get_historical_data(ticker)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Ignore the Streamlit warning for using st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Main title
st.title('Option pricing')

# User selected model from sidebar 
pricing_method = st.sidebar.radio('Please select option pricing method', options=[model.value for model in OPTION_PRICING_MODEL])

# Displaying specified model
st.subheader(f'Pricing method: {pricing_method}')

if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    # Parameters for Black-Scholes model
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    
    if st.button(f'Calculate option price for {ticker}'):
        with st.spinner('Fetching data...'):
            data = get_historical_data(ticker)
        
        if data is not None and not data.empty:
            st.write("Data fetched successfully:")
            st.write(data.tail())
            
            fig = Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot(fig)

            spot_price = Ticker.get_last_price(data, 'Adj Close')
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
            call_option_price = BSM.calculate_option_price('Call Option')
            put_option_price = BSM.calculate_option_price('Put Option')

            st.subheader(f'Call option price: {call_option_price:.2f}')
            st.subheader(f'Put option price: {put_option_price:.2f}')
        else:
            st.error("Unable to proceed with calculations due to data fetching error.")

elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    # Parameters for Monte Carlo simulation
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)

    if st.button(f'Calculate option price for {ticker}'):
        with st.spinner('Fetching data...'):
            data = get_historical_data(ticker)
        
        if data is not None and not data.empty:
            st.write("Data fetched successfully:")
            st.write(data.tail())
            
            fig = Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot(fig)

            spot_price = Ticker.get_last_price(data, 'Adj Close')
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
            MC.simulate_prices()

            MC.plot_simulation_results(num_of_movements)
            st.pyplot()

            call_option_price = MC.calculate_option_price('Call Option')
            put_option_price = MC.calculate_option_price('Put Option')

            st.subheader(f'Call option price: {call_option_price:.2f}')
            st.subheader(f'Put option price: {put_option_price:.2f}')
        else:
            st.error("Unable to proceed with calculations due to data fetching error.")

elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
    # Parameters for Binomial-Tree model
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
    number_of_time_steps = st.slider('Number of time steps', 5000, 100000, 15000)

    if st.button(f'Calculate option price for {ticker}'):
        with st.spinner('Fetching data...'):
            data = get_historical_data(ticker)
        
        if data is not None and not data.empty:
            st.write("Data fetched successfully:")
            st.write(data.tail())
            
            fig = Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot(fig)

            spot_price = Ticker.get_last_price(data, 'Adj Close')
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
            call_option_price = BOPM.calculate_option_price('Call Option')
            put_option_price = BOPM.calculate_option_price('Put Option')

            st.subheader(f'Call option price: {call_option_price:.2f}')
            st.subheader(f'Put option price: {put_option_price:.2f}')
        else:
            st.error("Unable to proceed with calculations due to data fetching error.")