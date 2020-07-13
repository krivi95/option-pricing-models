"""
Script testing functionalities of option_pricing package:
- Testing stock data fetching from Yahoo Finance using pandas-datareader
- Testing Black-Scholes option pricing model   
- Testing Binomial option pricing model   
- Testing Monte Carlo Simulation for option pricing   
"""

from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker

# Fetching the prices from yahoo finance
data = Ticker.get_historical_data('TSLA')
print(Ticker.get_columns(data))
print(Ticker.get_last_price(data, 'Adj Close'))
Ticker.plot_data(data, 'TSLA', 'Adj Close')

# Black-Scholes model testing
BSM = BlackScholesModel(100, 100, 365, 0.1, 0.2)
print(BSM.calculate_option_price('Call Option'))
print(BSM.calculate_option_price('Put Option'))

# Binomial model testing
BOPM = BinomialTreeModel(100, 100, 365, 0.1, 0.2, 15000)
print(BOPM.calculate_option_price('Call Option'))
print(BOPM.calculate_option_price('Put Option'))

# Monte Carlo simulation testing
MC = MonteCarloPricing(100, 100, 365, 0.1, 0.2, 10000)
MC.simulate_prices()
print(MC.calculate_option_price('Call Option'))
print(MC.calculate_option_price('Put Option'))
MC.plot_simulation_results(20)


