from option_pricing import BlackScholesModel, MonteCarloPricing, Ticker

# Fetching the prices from yahoo finance
data = Ticker.get_historical_data('TSLA')
print(Ticker.get_columns(data))
print(Ticker.get_last_price(data, 'Adj Close'))
Ticker.plot_data(data, 'TSLA', 'Adj Close')

# Black-Scholes model testing
BSM = BlackScholesModel(100, 100, 365, 0.1, 0.2)
print(BSM.calculate_option_price('Call Option'))
print(BSM.calculate_option_price('Put Option'))

# Monte Carlo simulation testing
MC = MonteCarloPricing(100, 100, 365, 0.1, 0.2, 10000)
MC.simulate_prices()
print(MC.calculate_option_price('Call Option'))
print(MC.calculate_option_price('Put Option'))
MC.plot_simulation_results(20)


