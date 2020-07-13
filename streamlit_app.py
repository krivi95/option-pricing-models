import streamlit as st
from enum import Enum
from datetime import datetime

from option_pricing import BlackScholesModel, MonteCarloPricing, Ticker

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Black Scholes Model'
    MONTE_CARLO = 'Monte Carlo Simulation'
    BINOMIAL = 'Binomial Model'

@st.cache
def get_historical_data(ticker):
    return Ticker.get_historical_data(ticker)

st.title('Option pricing')

pricing_method = st.sidebar.radio('Please select option pricing method', options=[model.value for model in OPTION_PRICING_MODEL])

st.subheader(f'Pricing method: {pricing_method}')

if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today())
    
    if st.button(f'Calculate option price for {ticker}'):
        data = get_historical_data(ticker)
        st.write(data.tail())
        Ticker.plot_data(data, ticker, 'Adj Close')
        st.pyplot()

        stop_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days
        # days_to_maturity = 365

        BSM = BlackScholesModel(stop_price, strike_price, days_to_maturity, risk_free_rate, sigma)
        call_option_price = BSM.calculate_option_price('Call Option')
        put_option_price = BSM.calculate_option_price('Put Option')

        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')

elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
    ticker = st.text_input('Ticker symbol', 'AAPL')
    strike_price = st.number_input('Strike price', 300)
    risk_free_rate = st.slider('Risk-free rate (%)', 0, 100, 10)
    sigma = st.slider('Sigma (%)', 0, 100, 20)
    exercise_date = st.date_input('Exercise date', min_value=datetime.today())
    number_of_simulations = st.slider('Number of simulations', 100, 100000, 10000)
    num_of_movements = st.slider('Number of price movement simulations to be visualized ', 0, int(number_of_simulations/10), 100)
    

    if st.button(f'Calculate option price for {ticker}'):
        data = get_historical_data(ticker)
        st.write(data.tail())
        Ticker.plot_data(data, ticker, 'Adj Close')
        st.pyplot()

        stop_price = Ticker.get_last_price(data, 'Adj Close') 
        risk_free_rate = risk_free_rate / 100
        sigma = sigma / 100
        days_to_maturity = (exercise_date - datetime.now().date()).days

        MC = MonteCarloPricing(stop_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
        MC.simulate_prices()

        MC.plot_simulation_results(num_of_movements)
        st.pyplot()

        call_option_price = MC.calculate_option_price('Call Option')
        put_option_price = MC.calculate_option_price('Put Option')

        st.subheader(f'Call option price: {call_option_price}')
        st.subheader(f'Put option price: {put_option_price}')

# st.checkbox('Show dataframe')



# option = st.selectbox(
#     'Which number do you like best?',
#      ['first', 'column'])

# 'You selected: ', option

# bar = st.progress(0)
# bar = st.progress(12)

# genre = st.radio(
#      "What's your favorite movie genre",
#      ('Comedy', 'Drama', 'Documentary'))

# title = st.text_input('Movie title', 'Life of Brian')