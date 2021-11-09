# Standart python imports
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
# colors generator
from itertools import cycle
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
import seaborn as sns

# Third party imports
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# Local package imports
from option_pricing import BlackScholesModel, MonteCarloPricing, BinomialTreeModel, Ticker
import tree_constructor as tc
import tree_plotter as tp

class OPTION_PRICING_MODEL(Enum):
    BLACK_SCHOLES = 'Modèle deBlack Scholes'
    MONTE_CARLO = 'Simulation de Monte Carlo'
    BINOMIAL = 'Modèle binoiale CRR'

class OPTION_PRICING_AMERICAN(Enum):
    MONTE_CARLO = 'Least Squares Monte Carlo Simulation'
    BINOMIAL = 'Modèle binoiale CRR'

class OPTION_TYPES(Enum):
    EUROPEAN = "Européenne"
    AMERICAN = "Américaine"

@st.cache
def get_historical_data(ticker):
    """Getting historical data for speified ticker and caching it with streamlit app."""
    return Ticker.get_historical_data(ticker)

def plot_yeild(stock):
    stock['log_returns'] = np.log(stock.iloc[:,0].pct_change(1) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1]})
    cols = [next(color_cycle) for _ in range(3)]
    for i, item in enumerate(stock.columns):
        stock[item].plot(title=item, color=cols[0], ax=axs[i][0], label="daily")
        stock[item].rolling(30).mean().plot(
            color=cols[1],
            ax=axs[i][0], label="monthly rolling mean", alpha=0.75)
        stock[item].rolling(90).mean().plot(
            color=cols[2],
            ax=axs[i][0], linestyle='--', label="yearly rolling mean")
        axs[i][0].legend(prop={'size': 8.5})
        sns.boxplot(y=stock[item], ax=axs[i][1], color=cols[0]).set(title=f"{item} variation")
    plt.tight_layout()
    plt.show()

    desc = stock.describe().drop('count')
    return pd.concat([desc.iloc[:2], pd.DataFrame(stock.skew(), columns=["skewness"]).T, 
                         pd.DataFrame(stock.kurt(), columns=["kurtosis"]).T, desc.iloc[3:]]).T


def black_scholes(K, T, S, r, sigma, opttype='C'):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opttype == 'C':
        price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price

def simulate_gbm(s_0, r, sigma, T, N, n_sims, random_seed=42, antithetic_var=True):
    np.random.seed(random_seed)
    # time increment
    dt = T/N
    # Brownian
    if antithetic_var:
        dW_ant = np.random.normal(scale=np.sqrt(dt), size=(N, int(n_sims/2)))
        dW = np.concatenate((dW_ant, -dW_ant), axis=1)
    else:
        dW = np.random.normal(scale=np.sqrt(dt), size=(N, n_sims))
    # simulate the evolution of the process
    S_t = s_0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * dW, axis=0))
    S_t[:, 0] = s_0
    return S_t

def lsmc_american_option(S_0, K, T, N, r, sigma, n_sims, option_type, poly_degree, random_seed=42):
    dt = T / N
    discount_factor = np.exp(-r * dt)
    # Simuler le mouvement du prix de l'actif sous-jacent par le GBM
    gbm_simulations = simulate_gbm(s_0=S_0, r=r, sigma=sigma, n_sims=n_sims, T=T, N=N, random_seed=random_seed)
    
    # Calculer la matrice des payoffs
    payoff_matrix_call = np.maximum(
        gbm_simulations - K, np.zeros_like(gbm_simulations))

    payoff_matrix_put = np.maximum(
        K - gbm_simulations, np.zeros_like(gbm_simulations))
    
    option_premium = []
    for payoff_matrix in (payoff_matrix_call, payoff_matrix_put):
        # Définir la matrice de valeurs et remplir la dernière colonne (temps T)
        value_matrix = np.zeros_like(payoff_matrix)
        value_matrix[:, -1] = payoff_matrix[:, -1]

        # Calculer itérativement la valeur de continuation et le vecteur de valeur dans le temps donné
        for t in range(T - 1, 0, -1):
            regression = np.polyfit(
                gbm_simulations[:, t], value_matrix[:, t + 1] * discount_factor, poly_degree)
            continuation_value = np.polyval(regression, gbm_simulations[:, t])
            # Si le gain était supérieur à la valeur attendue de la continuation, nous définissons la valeur sur le gain. 
            # Sinon, nous la définissons sur la valeur actualisée d'un pas en avant
            value_matrix[:, t] = np.where(payoff_matrix[:, t] > continuation_value,
                                          payoff_matrix[:, t],
                                          value_matrix[:, t + 1] * discount_factor)

        # Calculer la prime de l'option
        option_premium.append(np.mean(value_matrix[:, 1] * discount_factor))
    return option_premium[0], option_premium[1], gbm_simulations

def plot_simulation_results(S, K, N, num_of_movements, ax):
    """Plots specified number of simulated price movements."""
    ax.plot(S[:,0:num_of_movements])
    ax.axhline(K, c='k', xmin=0, xmax=N, label='Strike Price')
    ax.set_xlim([0, S.shape[0]])
    ax.set_ylabel('Simulated price movements')
    ax.set_xlabel('Days in future')
    ax.set_title(f'First {num_of_movements}/{S.shape[1]} Random Price Movements')
    ax.legend(loc='best')

def normal(mean, std, ax=plt, histmax=False, color="crimson"):
    """
    histmax : set it to ax.get_ylim()[1] when stat="probability" in sns.histplot
    """
    x = np.linspace(mean-4*std, mean+4*std, 200)
    p = norm.pdf(x, mean, std)
    if histmax:
        p = p*histmax/max(p)
    z = ax.plot(p, x, color, linewidth=1.5, linestyle='--', label='Theoretical distribution')

def american_CRR(K,T,S0,r,N,sigma,opttype='P'):
    #precompute values
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    q = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)
    
    # initialise stock prices at maturity
    S = S0 * d**(np.arange(N,-1,-1)) * u**(np.arange(0,N+1,1))
        
    # option payoff 
    if opttype == 'P':
        C = np.maximum(0, K - S)
    else:
        C = np.maximum(0, S - K)
    
    # backward recursion through the tree
    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        C[:i+1] = disc * (q*C[1:i+2] + (1-q)*C[0:i+1])
        C = C[:-1]
        if opttype == 'P':
            C = np.maximum(C, K - S)
        else:
            C = np.maximum(C, S - K)    
    return C[0]

# Main title
st.title('Pricing des options')



option_type = st.sidebar.selectbox("Veuillez sélectionner le type d'option", [model.value for model in OPTION_TYPES])
if option_type == OPTION_TYPES.EUROPEAN.value:
    # User selected model from sidebar 
    pricing_method = st.sidebar.radio("Veuillez sélectionner la méthode de Pricing", options=[model.value for model in OPTION_PRICING_MODEL])

    # Displaying specified model
    st.subheader(f'Méthode de Pricing : {pricing_method}')

    if pricing_method == OPTION_PRICING_MODEL.BLACK_SCHOLES.value:
        # Parameters for Black-Scholes model
        ticker = st.text_input("Choisir l'action sous jacente", 'AAPL')
        strike_price = st.number_input("Prix d'exercice", 200)
        risk_free_rate = st.slider('Taux annuel sans risque (%)', 0, 100, 10)
        sigma = st.slider('Volatilité implicite : Sigma (%)', 0, 100, 20)
        exercise_date = st.date_input("Date de maturité", min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        
        if st.button(f"Calculer le prix de l'option pour {ticker}"):
            # Getting data for selected ticker
            data = get_historical_data(ticker)
            st.write(data.tail())
            plot_yeild(data[['Adj Close']])
            # Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot()

            # Formating selected model parameters
            spot_price = Ticker.get_last_price(data, 'Adj Close') 
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            # Calculating option price
            BSM = BlackScholesModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma)
            call_option_price = BSM.calculate_option_price('Call Option')
            put_option_price = BSM.calculate_option_price('Put Option')

            # Displaying call/put option price
            st.subheader(f"Prix du Call : {call_option_price}")
            st.subheader(f"Prix du Put : {put_option_price}")
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            eur_bs_call, eur_bs_put = [], []

            S_prices = np.linspace(min(strike_price, spot_price)*0.25, max(strike_price, spot_price)*2, 100)
            call_payoff = np.maximum(S_prices - strike_price, 0) # payoff of the option
            put_payoff = np.maximum(strike_price - S_prices, 0)

            for S_price in S_prices:
                # BSM = BlackScholesModel(S_price, strike_price, days_to_maturity, risk_free_rate, sigma)
                
                # eur_bs_call.append(BSM.calculate_option_price('Call Option'))
                # eur_bs_put.append(BSM.calculate_option_price('Put Option'))
                                
                eur_bs_call.append(black_scholes(strike_price, days_to_maturity/365, S_price, risk_free_rate, sigma, opttype='C'))
                eur_bs_put.append(black_scholes(strike_price, days_to_maturity/365, S_price, risk_free_rate, sigma, opttype='P'))

            axs[0].plot(S_prices, eur_bs_call, label="Prix de l'option", linewidth=4)
            axs[0].plot(S_prices, call_payoff, label='Payoff', color='k', linestyle='--')
            axs[0].axvline(x=strike_price, color='g', linestyle='--', label='Strike')
            axs[0].axvline(x=spot_price, color='r', linestyle=':', label='S_0')
            axs[0].set_xlabel("S_t")
            axs[0].set_ylabel("Prix de l'option")
            axs[0].set_title("Call Européen")
            axs[0].legend(loc='upper left')

            axs[1].plot(S_prices, eur_bs_put, label="Prix de l'option", linewidth=4)
            axs[1].plot(S_prices, put_payoff, label='Payoff', color='k', linestyle='--')
            axs[1].axvline(x=strike_price, color='g', linestyle='--', label='Strike')
            axs[1].axvline(x=spot_price, color='r', linestyle=':', label='S_0')
            axs[1].set_xlabel("S_t")
            axs[1].set_xlabel("Prix du sous-jacent")
            axs[1].set_ylabel("Prix de l'option")
            axs[1].set_title("Put Européen")
            axs[1].legend(loc='upper left')

            plt.tight_layout()
            plt.show()
            st.pyplot()

    elif pricing_method == OPTION_PRICING_MODEL.MONTE_CARLO.value:
        # Parameters for Monte Carlo simulation
        ticker = st.text_input("Choisir l'action sous jacente", 'AAPL')
        strike_price = st.number_input("Prix d'exercice", 200)
        risk_free_rate = st.slider('Taux annuel sans risque (%)', 0, 100, 10)
        sigma = st.slider('Volatilité implicite : Sigma (%)', 0, 100, 20)
        exercise_date = st.date_input("Date de maturité", min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        number_of_simulations = st.slider('Nombre de simulations', 100, 100000, 10000)
        num_of_movements = st.slider('Nombre de chemin de prix à visualiser', 0, int(number_of_simulations/5), 1000)

        if st.button(f"Calculer le prix de l'option pour {ticker}"):
            # Getting data for selected ticker
            data = get_historical_data(ticker)
            st.write(data.tail())
            plot_yeild(data[['Adj Close']])
            # Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot()

            # Formating simulation parameters
            spot_price = Ticker.get_last_price(data, 'Adj Close') 
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            # ESimulating stock movements
            MC = MonteCarloPricing(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_simulations)
            MC.simulate_prices()

            # Visualizing Monte Carlo Simulation
            MC.plot_simulation_results(num_of_movements)
            st.pyplot()

            # Calculating call/put option price
            call_option_price = MC.calculate_option_price('Call Option')
            put_option_price = MC.calculate_option_price('Put Option')

            # Displaying call/put option price
            st.subheader(f'Prix du Call : {call_option_price}')
            st.subheader(f'Prix du Put : {put_option_price}')

    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        # Parameters for Binomial-Tree model
        ticker = st.text_input("Choisir l'action sous jacente", 'AAPL')
        strike_price = st.number_input("Prix d'exercice", 150)
        risk_free_rate = st.slider('Taux annuel sans risque (%)', 0, 100, 10)
        sigma = st.slider('Volatilité implicite : Sigma (%)', 0, 100, 20)
        exercise_date = st.date_input("Date de maturité", min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        number_of_time_steps = st.slider('Nombre de pas dans le temps', 500, 100000, 15000)
        if (exercise_date - datetime.now().date()).days < 16:
            st.subheader('Légende:')
            st.markdown("✅ Arbre des prix de l'action : En noir")
            call = st.checkbox('Arbre des prix du Call : En bleu')
            put = st.checkbox('Arbre des prix du Put : En rouge')

        if st.button(f"Calculer le prix de l'option pour {ticker}"):
             # Getting data for selected ticker
            data = get_historical_data(ticker)
            st.write(data.tail())
            plot_yeild(data[['Adj Close']])
            # Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot()

            # Formating simulation parameters
            spot_price = Ticker.get_last_price(data, 'Adj Close') 
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            # Calculating option price
            if days_to_maturity < 16:
                dp = 6
                dT = days_to_maturity / number_of_time_steps                             
                u = np.exp(sigma * np.sqrt(dT))
                trees = tc.Calculate(spot_price, strike_price, days_to_maturity, risk_free_rate, u, dp)

                # plot stock tree
                tp.plot_stock_lattice(trees.stock_prices())

                # plot exercise price reference
                plt.axhline(y=strike_price, color='k', linestyle="dashed", label="Strike")
                plt.annotate(strike_price, (0, strike_price), textcoords="offset points",
                             xytext=(0,-15), ha='center')
                plt.legend()

                # plot options trees
                if call:
                    tp.plot_option_lattice(trees.call_values())
                if put:
                    tp.plot_option_lattice(trees.put_values(), 'ro-')
                st.pyplot(plt)
                
                call_option_price = round(trees.call_values()[-1][0], 4)
                put_option_price = round(trees.put_values()[-1][0], 4)
            else:        
                BOPM = BinomialTreeModel(spot_price, strike_price, days_to_maturity, risk_free_rate, sigma, number_of_time_steps)
                call_option_price = BOPM.calculate_option_price('Call Option')
                put_option_price = BOPM.calculate_option_price('Put Option')

            # Displaying call/put option price
            st.subheader(f"Prix du Call : {call_option_price}")
            st.subheader(f"Prix du Put : {put_option_price}")
else:
    pricing_method = st.sidebar.radio('Veuillez sélectionner la méthode de Pricing', options=[model.value for model in OPTION_PRICING_AMERICAN])

    # Displaying specified model
    st.subheader(f'la méthode de Pricing : {pricing_method}')
    
    if pricing_method == OPTION_PRICING_AMERICAN.MONTE_CARLO.value:
        # Parameters for Monte Carlo simulation
        ticker = st.text_input("Choisir l'action sous jacente", 'AAPL')
        strike_price = st.number_input("Prix d'exercice", 200)
        risk_free_rate = st.slider('Taux annuel sans risque (%)', 0, 100, 10)
        sigma = st.slider('Volatilité implicite : Sigma (%)', 0, 100, 20)
        exercise_date = st.date_input("Date de maturité", min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        number_of_simulations = st.slider('Nombre de simulations', 100, 100000, 10000)
        num_of_movements = st.slider('Nombre de chemin de prix à visualiser ', 0, int(number_of_simulations/5), 1000)
        number_of_time_steps = st.slider('Nombre de pas dans le temps', 500, 100000, 15000)
        
        if st.button(f"Calculer le prix de l'option pour {ticker}"):
            # Getting data for selected ticker
            data = get_historical_data(ticker)
            st.write(data.tail())
            plot_yeild(data[['Adj Close']])
            # Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot()

            # Formating simulation parameters
            spot_price = Ticker.get_last_price(data, 'Adj Close') 
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days
            
            call_option_price, put_option_price, GBM = lsmc_american_option(spot_price, strike_price, days_to_maturity, number_of_time_steps, risk_free_rate, 
                                                       sigma, n_sims=number_of_simulations, option_type='P', poly_degree=4)
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
            plot_simulation_results(S=GBM, K=strike_price, N=number_of_time_steps, num_of_movements=num_of_movements, ax=axs[0])
            disp = sns.histplot(y=GBM[-1], stat="density", kde=True, color ='blue', ax=axs[1], label='Obesrved distribution')
            mean, std = np.exp(np.log(GBM[-1]).mean()), np.log(GBM[-1]).std()
            x = np.linspace(mean-4*std, mean+4*std, 200)
            from scipy.stats import lognorm
            p = lognorm.pdf(x,std,0,mean)
            axs[1].plot(p, x, "crimson", linewidth=1.5, linestyle='--', label='Theoretical distribution')
            # normal(GBM[-1].mean(), GBM[-1].std(), ax=axs[1])
            axs[1].legend(prop={'size': 9}, loc="upper left")
            axs[1].set_title(f"Geometric Brownian motion Distribution")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)

            # Displaying call/put option price
            st.subheader(f"Prix du Call : {call_option_price}")
            st.subheader(f"Prix du Put : {put_option_price}")

    elif pricing_method == OPTION_PRICING_MODEL.BINOMIAL.value:
        # Parameters for Binomial-Tree model
        ticker = st.text_input("Choisir l'action sous jacente", 'AAPL')
        strike_price = st.number_input("Prix d'exercice", 150)
        risk_free_rate = st.slider('Taux annuel sans risque (%)', 0, 100, 10)
        sigma = st.slider('Volatilité implicite : Sigma (%)', 0, 100, 20)
        exercise_date = st.date_input("Date de maturité", min_value=datetime.today() + timedelta(days=1), value=datetime.today() + timedelta(days=365))
        number_of_time_steps = st.slider('Nombre de pas dans le temps', 500, 100000, 15000)
        if (exercise_date - datetime.now().date()).days < 16:
            st.subheader('Légende:')
            st.markdown("✅ Arbre des prix de l'action : En noir")
            call = st.checkbox('Arbre des prix du Call : En bleu')
            put = st.checkbox('Arbre des prix du Put : En rouge')

        if st.button(f"Calculer le prix de l'option pour {ticker}"):
             # Getting data for selected ticker
            data = get_historical_data(ticker)
            st.write(data.tail())
            plot_yeild(data[['Adj Close']])
            # Ticker.plot_data(data, ticker, 'Adj Close')
            st.pyplot()

            # Formating simulation parameters
            spot_price = Ticker.get_last_price(data, 'Adj Close') 
            risk_free_rate = risk_free_rate / 100
            sigma = sigma / 100
            days_to_maturity = (exercise_date - datetime.now().date()).days

            # Calculating option price
            if days_to_maturity < 16:
                dp = 6
                dT = days_to_maturity / number_of_time_steps                             
                u = np.exp(sigma * np.sqrt(dT))
                trees = tc.Calculate(spot_price, strike_price, days_to_maturity, risk_free_rate, u, dp, opttype='Americ')

                # plot stock tree
                tp.plot_stock_lattice(trees.stock_prices())

                # plot exercise price reference
                plt.axhline(y=strike_price, color='k', linestyle="dashed", label="Strike")
                plt.annotate(strike_price, (0, strike_price), textcoords="offset points",
                             xytext=(0,-15), ha='center')
                plt.legend()

                # plot options trees
                if call:
                    tp.plot_option_lattice(trees.call_values())
                if put:
                    tp.plot_option_lattice(trees.put_values(), 'ro-')
                st.pyplot(plt)
                
                call_option_price = round(trees.call_values()[-1][0], 4)
                put_option_price = round(trees.put_values()[-1][0], 4)
            else:                        
                call_option_price = american_CRR(strike_price,days_to_maturity,spot_price,risk_free_rate,number_of_time_steps,sigma,opttype='C')
                put_option_price = american_CRR(strike_price,days_to_maturity,spot_price,risk_free_rate,number_of_time_steps,sigma,opttype='P')

            # Displaying call/put option price
            st.subheader(f"Prix du Call : {call_option_price}")
            st.subheader(f"Prix du Put : {put_option_price}")
            
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
            amer_crr_call, amer_crr_put = [], []

            S_prices = np.linspace(min(strike_price, spot_price)*0.5, max(strike_price, spot_price)*1.5, 50)
            call_payoff = np.maximum(S_prices - strike_price, 0) # payoff of the option
            put_payoff = np.maximum(strike_price - S_prices, 0)
            for S_price in S_prices:
                call = american_CRR(strike_price,days_to_maturity,S_price,risk_free_rate,number_of_time_steps,sigma,opttype='C')
                put = american_CRR(strike_price,days_to_maturity,S_price,risk_free_rate,number_of_time_steps,sigma,opttype='P')
                amer_crr_call.append(call)
                amer_crr_put.append(put)

            axs[0].plot(S_prices, amer_crr_call, label="Prix de l'option", linewidth=4)
            axs[0].plot(S_prices, call_payoff, label='Payoff', color='k', linestyle='--')
            axs[0].axvline(x=strike_price, color='g', linestyle='--', label='Strike')
            axs[0].axvline(x=spot_price, color='r', linestyle=':', label='S_0')
            axs[0].set_title("Call Américaine")
            axs[0].legend(loc='upper left')

            axs[1].plot(S_prices, amer_crr_put, label="Prix de l'option", linewidth=4)
            axs[1].plot(S_prices, put_payoff, label='Payoff', color='k', linestyle='--')
            axs[1].axvline(x=strike_price, color='g', label='Strike')
            axs[1].axvline(x=spot_price, color='r', linestyle=':', label='S_0')
            axs[1].set_xlabel("Prix du sous-jacent")
            axs[1].set_title("Put Américaine")
            axs[1].legend(loc='upper left')
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)