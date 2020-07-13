# Third party imports
import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

# Local package imports
from .base import OptionPricingModel


class MonteCarloPricing(OptionPricingModel):
    """ 
    Class implementing calculation for European option price using Monte Carlo Simulation.
    We simulate underlying asset price on expiry date using random stochastic process - Brownian motion.
    For the simulation generated prices at maturity, we calculate and sum up their payoffs, average them and discount the final value.
    That value represents option price
    """

    def __init__(self, underlying_spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations):
        """
        Initializes variables used in Black-Scholes formula .

        underlying_spot_price: current stock or other underlying spot price
        strike_price: strike price for option cotract
        days_to_maturity: option contract maturity/exercise date
        risk_free_rate: returns on risk-free assets (assumed to be constant until expiry date)
        sigma: volatility of the underlying asset (standard deviation of asset's log returns)
        number_of_simulations: number of potential random underlying price movements 
        """
        # Parameters for Brownian process
        self.S_0 = underlying_spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = sigma 

        # Parameters for simulation
        self.N = number_of_simulations
        self.num_of_steps = days_to_maturity
        self.dt = self.T / self.num_of_steps

    def simulate_prices(self):
        """
        Simulating price movement of underlying prices using Brownian random process.
        Saving random results.
        """
        np.random.seed(20)
        self.simulation_results = None

        # Initializing price movements for simulation: rows as time index and columns as different random price movements.
        S = np.zeros((self.num_of_steps, self.N))        
        # Starting value for all price movements is the current spot price
        S[0] = self.S_0

        for t in range(1, self.num_of_steps):
            # Random values to simulate Brownian motion (Gaussian distibution)
            Z = np.random.standard_normal(self.N)
            # Updating prices for next point in time 
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + (self.sigma * np.sqrt(self.dt) * Z))

        self.simulation_results_S = S

    def _calculate_call_option_price(self): 
        """
        Call option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Call option payoff (it's exercised only if the price at expiry date is higher than a strike price): max(S_t - K, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.simulation_results_S[-1] - self.K, 0))
    

    def _calculate_put_option_price(self): 
        """
        Put option price calculation. Calculating payoffs for simulated prices at expiry date, summing up, averiging them and discounting.   
        Put option payoff (it's exercised only if the price at expiry date is lower than a strike price): max(K - S_t, 0)
        """
        if self.simulation_results_S is None:
            return -1
        return np.exp(-self.r * self.T) * 1 / self.N * np.sum(np.maximum(self.K - self.simulation_results_S[-1], 0))
       

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12,8))
        plt.plot(self.simulation_results_S[:,0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.num_of_steps, label='Strike Price')
        plt.xlim([0, self.num_of_steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        plt.show()