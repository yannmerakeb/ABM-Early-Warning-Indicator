import pandas as pd
import numpy as np

from typing_extensions import Optional

from data_preprocessor import Data
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2, norm
from datetime import datetime


class Return(Data):
    '''def __init__(self, dict_data : dict):
        # Get the index and stock data
        self.index = dict_data['index']
        self.stocks = dict_data['stocks']

        # Get the names of the indexes and stocks
        self.index_names = self.index.columns[1]
        self.stock_names = list(self.stocks.keys())'''

    def cumulative_returns(self, index: bool):
        if index:
            ratio = (self.index / self.index[0]).astype(float)
            return np.log(ratio)
        else:
            self.stock_cum_returns_dict = {}
            for stock in self.stock_names:
                ratio = (self.stocks[stock] / self.stocks[stock][0]).astype(float)
                self.stock_cum_returns_dict[stock] = np.log(ratio)
            return self.stock_cum_returns_dict

    def returns(self, index: bool):
        '''
        Compute the returns of the index or the stocks
        Args:
            index (bool) : True if the index, False if the stocks
        Return:
            np.array or dict : returns
        '''
        if index:
            return self.cumulative_returns(index).diff()[1:].to_numpy().reshape(-1,1)
        else:
            cum_returns = self.cumulative_returns(index=False)

            self.stock_returns_dict = {}
            for stock in self.stock_names:
                self.stock_returns_dict[stock] = cum_returns[stock].diff()[1:].to_numpy().reshape(-1,1)
            return self.stock_returns_dict

    @property
    def avg_daily_return(self):
        '''
        OLS regression (cumulative_returns = cst + avg_daily_return * t => y = beta[0] + beta[1] * X)
        Return:
            np.array : beta
        '''
        """cst_ones = np.ones((len(self.index[1:]), 1))
        t = np.arange(len(self.index[1:]))"""

        cst_ones = np.ones((len(self.index[1:]), 1))
        #t = np.arange(1, len(self.index[1:]) + 1)
        t = np.arange(1, len(self.index))

        X = np.column_stack((cst_ones, t))
        y = self.cumulative_returns(index=True)[1:]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta

    @property
    def detrended_prices(self) -> dict:
        '''
        Detrend stock prices
        Return:
            dict : Detrended prices = prices * exp(-avg_daily_return * t)
        '''
        # Number of days, including 0 for the first day (not detrended)
        t = np.arange(len(self.index))
        # t = self.index.index.to_numpy().reshape(-1, 1)
        detrend_factor = np.exp(-self.avg_daily_return[1] * t)

        self.detrended_prices_dict = {}
        for stock in self.stock_names:
            self.detrended_prices_dict[stock] = self.stocks[stock] * detrend_factor
        return self.detrended_prices_dict

class SentimentIndex(Data):
    def __init__(self, dict_data: dict, EMA_window: int = 100):
        # Exponential Moving Average time window
        super().__init__(dict_data)
        self.L = EMA_window
        # Weight used for the EMA
        self.W = 2 / (self.L + 1)

        self.detrended_prices = Return(self.dict_data).detrended_prices

    def _EMA_init(self) -> dict:
        '''
        Initialize the EMA, the first value is the SMA
        Return:
            dict : initialized EMA

        '''
        self.EMA_dict = {}

        for stock in self.stock_names:
            '''ema_data = pd.concat([self.stocks[stock]['Date'], pd.DataFrame(index=self.stocks[stock].index, columns=[stock])], axis=1)
            ema_data.iloc[self.L] = np.mean(self.stocks[stock].iloc[:self.L, 1])
            self.EMA_dict[stock] = ema_data
            ema_data = pd.concat([self.stocks[stock]['Date'], pd.DataFrame(index=self.stocks[stock].index, columns=[stock])], axis=1)
            # ema_data.iloc[self.L] = np.mean(self.stocks[stock].iloc[:self.L, 1])'''

            ema_data = np.full(len(self.detrended_prices[stock]), np.nan)
            ema_data[self.L] = np.mean(self.detrended_prices[stock][:self.L])
            self.EMA_dict[stock] = ema_data

        return self.EMA_dict

    @property
    def pessimistic_state(self) -> dict:
        '''
        Compute the EMA by stock, and then the pessimistic state (binary)
        Return:
            dict : pessimistic state by stock
        '''
        self.EMA_dict = self._EMA_init()
        self.pessimistic_state_dict = {}

        for stock in self.stock_names:
            for row in range(self.L + 1, len(self.stocks[stock])):
                # Exponential Moving Average
                ema = self.W * self.detrended_prices[stock][row] + (1 - self.W) * self.EMA_dict[stock][row - 1]
                self.EMA_dict[stock][row] = ema

            # Pessimistic state
            pessimistic_state = self.detrended_prices[stock] < self.EMA_dict[stock]
            self.pessimistic_state_dict[stock] = pessimistic_state

        return self.pessimistic_state_dict

    @property
    def sentiment_index(self) -> pd.DataFrame:
        '''
        Compute the sentiment index and the daily changes
        Return:
            pd.DataFrame : sentiment index
        '''
        pessimistic_states = self.pessimistic_state
        sentiment_index_list = []

        for date_index in range(len(self.index)):
            agg_pessimistic_state = 0
            for stock in self.stock_names:
                agg_pessimistic_state += pessimistic_states[stock][date_index]

            sentiment_index = agg_pessimistic_state / len(self.stock_names)
            sentiment_index_list.append(sentiment_index)

        sentiment_index_df = pd.concat([self.dates, pd.DataFrame(sentiment_index_list, columns=['Sentiment Index'])], axis=1)
        sentiment_index_df = sentiment_index_df.iloc[self.L:,:]

        return sentiment_index_df

    def autocorrelation(self):
        '''
        Compute the autocorrelation function of the sentiment index
        Return:
            plt
        '''
        # Maximum Likelihood Estimation
        likelihood_obj = Likelihood(self.dict_data)
        e1, e2, b = likelihood_obj.MLE('normal', 0.95)[0]

        # Range of lags
        t = np.arange(100)

        # Autocorrelation function
        ac_fun = np.exp(-b * (e1 + e2) * t)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(t, ac_fun, label='Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function of the Sentiment Index')
        plt.legend()
        plt.show()

class Likelihood(Data):

    def theoretical_vs_empirical(self, distribution: str):
        '''
        Compare the theoretical distribution to the empirical distribution
        Args:
            distribution (str): Theoretical distribution to compare ('beta' or 'normal')
        '''
        # Sentiment index
        sentiment_index = SentimentIndex(self.dict_data).sentiment_index.iloc[:, 1].to_numpy()

        # Parameters
        x = np.linspace(0.05, 0.95, 100)
        if distribution == 'beta':
            e1, e2 = self.MLE('beta')
            y = beta.pdf(x, e1, e2)
        elif distribution == 'normal':
            e1, e2, b = self.MLE('normal')
            z_bar = e1 / (e1 + e2)

            '''# Calculate mu(x) and sigma(x) for each x
            mu = x + (e1 + e2) * (z_bar - x) * b
            sigma = np.sqrt(2 * b * (1 - x) * x)

            # Theoretical PDF as per your model
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)'''

            '''for z_t in sentiment_index:
                # Calcul des paramètres conditionnels
                mu = z_t + (e1 + e2) * (z_bar - z_t) * b
                sigma = np.sqrt(2 * b * (1 - z_t) * z_t)

                # Distribution normale conditionnelle
                y = norm.pdf(x, loc=mu, scale=sigma)
                plt.plot(x, y, label=f"$z_t={z_t:.2f}$")

            plt.show()'''

            conditional_densities = []
            x = np.linspace(0.05, 0.95, 1000)
            # Calcul des densités conditionnelles pour chaque z_t
            for z_t in x:
                mu = z_t + (e1 + e2) * (z_bar - z_t) * b
                sigma = np.sqrt(2 * b * (1 - z_t) * z_t)
                density = norm.pdf(x, loc=mu, scale=sigma)
                conditional_densities.append(density)

            # Moyenne des densités conditionnelles (pondération uniforme)
            average_density = np.mean(conditional_densities, axis=0)

            plt.plot(x, average_density, color="red", linestyle="--", label="PDF Théorique Moyenne (Parabole)")
            plt.show()

        else:
            raise ValueError("Distribution not supported")

        # Calculate the histogram data
        counts, bins = np.histogram(sentiment_index, bins=30, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plot the points at the tops of the histogram bars
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, counts, color='blue', label='Empirical')
        plt.plot(x, y, color='red', label='Theoretical')
        plt.xlabel('Sentiment Index')
        plt.ylabel('Density')
        plt.title(f'Theoretical vs Empirical Distribution of the Sentiment Index ({distribution.capitalize()})')
        plt.legend()
        plt.show()

    # Unconditional distribution of the sentiment index
    def neg_beta_log_likelihood(self, params: tuple, sentiment_index: np.ndarray) -> float:
        '''
        Compute the beta log likelihood
        Return:
            float : beta log likelihood
        '''
        # Parameters
        e1, e2 = params

        '''# Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)
        sentiment_index = sentiment_index_obj.sentiment_index.iloc[:,1].to_numpy()'''

        # Number of observations, which is the number of days in the sentiment index
        n = len(sentiment_index)

        # Log likelihood
        sum_inv_B = n * (gammaln(e1 + e2) - gammaln(e1) - gammaln(e2))
        sum_ln_x = (e1 - 1) * np.sum(np.log(sentiment_index))
        sum_one_minus_ln_x = (e2 - 1) * np.sum(np.log(1 - sentiment_index))
        ll = sum_inv_B + sum_ln_x + sum_one_minus_ln_x

        return -ll

    # Conditional distribution of the sentiment index
    def neg_normal_log_likelihood(self, params: tuple, sentiment_index: np.ndarray) -> float:
        '''
        Compute the Normal log likelihood
        Args:
            params (tuple) : Normal parameters
            sentiment_index (np.ndarray) : Sentiment index
        Return:
            float : Normal log likelihood
        '''
        # Parameters
        e1, e2, b = params

        # Mean and standard deviation
        z_bar = e1 / (e1 + e2)
        mu = (sentiment_index + (e1 + e2) * (z_bar - sentiment_index) * b)[:-1]
        #mu = (sentiment_index + (e1 - (e1 + e2) * sentiment_index) * b)[:-1]
        sigma = np.sqrt(2 * b * (1 - sentiment_index) * sentiment_index)[:-1]

        # Log likelihood
        '''ll = (- 0.5 * np.sum(np.log(2 * np.pi * sigma[:-1] ** 2))
              - 0.5 * np.sum(((sentiment_index[1:] - mu[:-1]) / sigma[:-1]) ** 2))'''
        ll = (- 0.5 * np.sum(np.log(2 * np.pi * sigma ** 2))
              - 0.5 * np.sum(((sentiment_index[1:] - mu) / sigma) ** 2))

        """
        n = len(sentiment_index)
        z = sentiment_index

        ll = 0.0
        count_valid = 0  # Pour compter le nb de points utilisés

        for t in range(n - 1):
            zt = z[t]
            ztp1 = z[t + 1]

            # -- Sauter les points proches de 0 ou 1 --
            if (zt <= 1e-5) or (zt >= 1 - 1e-5):
                continue

            # Calcul mu
            mu_t = zt + (e1 - (e1 + e2) * zt) * b

            # Calcul var (avec plancher)
            var_t = 2.0 * b * zt * (1.0 - zt)
            if var_t < 1e-12:
                var_t = 1e-12

            # Log PDF gaussienne
            diff = ztp1 - mu_t

            # -0.5 ln(2 pi var) - (diff^2)/(2 var)
            log_pdf = -0.5 * np.log(2.0 * np.pi * var_t) \
                      - 0.5 * (diff ** 2 / var_t)

            ll += log_pdf
            count_valid += 1

        # Si vous voulez une moyenne par point, libre à vous,
        # mais traditionnellement on renvoie -sum(loglik).
        return -ll if count_valid > 0 else 1e10
        """

        return -ll

    # Maximum Likelihood Estimation
    def MLE(self, distribution: str, drop_extreme: float = None, start: int = None, end: int = None) -> tuple:
        '''
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_extreme (float) : Drop sentiment index values above this threshold (optional)
            start (int) : Start date index
            end (int) : End date index
        Return:
            tuple : MLE parameters
        '''
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)

        if start is None or end is None:
            sentiment_index = sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy()
        else:
            sentiment_index = sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy()[start:end]

        # Drop extreme sentiment index values if specified
        if drop_extreme:
            sentiment_index = sentiment_index[sentiment_index <= drop_extreme]

        # Clip the sentiment index to avoid log(0) and log(1)
        epsilon = 1e-3
        sentiment_index = np.clip(sentiment_index, epsilon, 1 - epsilon)

        # Optimization options
        options = {'maxiter': 100000, 'ftol': 1e-10, 'gtol': 1e-10, 'xtol': 1e-10}

        # Maximum Likelihood Estimation
        if distribution.lower() == 'beta':
            params_init = np.array([1, 1])
            bounds = ((1e-10, None), (1e-10, None))
            MLE = minimize(self.neg_beta_log_likelihood, params_init, args=(sentiment_index,),
                           bounds=bounds, method='L-BFGS-B', options=options)
            return MLE.x

        elif distribution.lower() == 'normal':
            params_init = np.array([1, 1, 1])
            bounds = ((1e-10, None), (1e-10, None), (1e-10, None))
            MLE = minimize(self.neg_normal_log_likelihood, params_init, args=(sentiment_index,),
                           bounds=bounds, method='L-BFGS-B', options=options)

            e1, e2, b = MLE.x
            z_bar = e1 / (e1 + e2)
            # moments = (sentiment_index + (e1 + e2) * (z_bar - sentiment_index) * b,
            #           np.sqrt(2 * b * (1 - sentiment_index) * sentiment_index) ** 2)

            return MLE.x
            # return MLE.x, moments

        else:
            raise ValueError("Distribution not supported")

class MonteCarlo(Data):
    def simulation(self, num_simulations: int = 1000, num_days: Optional[int] = None) -> np.ndarray:
        '''
        Perform Monte Carlo simulation based on the Normal distribution and parameters
        Args:
            num_simulations (int) : Number of simulations to run
            num_days (int) : Number of days to simulate
        Return:
            np.ndarray : Simulated paths
        '''
        likelihood = Likelihood(self.dict_data)

        # Number of days
        if not isinstance(num_days, int):
            num_days = len(self.index)

        # Parameters
        e1, e2, b = likelihood.MLE('normal', 0.95)

        # Initialize the simulated sentiment index with zeros
        simulated_sentiment_index = np.zeros((num_days, num_simulations))

        # Simulate the sentiment index
        for simul in range(num_simulations):
            for day in range(1, num_days):
                while not (simulated_sentiment_index[day, simul] > 0 and simulated_sentiment_index[day, simul] < 1):
                    simulated_sentiment_index[day, simul] = self._individual_simulation(simulated_sentiment_index[day - 1, simul],
                                                                                        e1, e2, b)

        return simulated_sentiment_index

    def _individual_simulation(self, z_t, e1, e2, b) -> float:
        '''
        Compute the next sentiment index value based on the Normal distribution
        Args:
            z_t (float) : Current sentiment index value
            e1, e2, b (float) : Parameters used to compute the mean and standard deviation of the Normal distribution
        '''
        # Compute the mean and standard deviation of the Normal distribution
        z_bar = e1 / (e1 + e2)
        mu = z_t + (e1 + e2) * (z_bar - z_t) * b
        vol = np.sqrt(2 * b * (1 - z_t) * z_t)

        # Generate a random Brownian increment with size 1
        lambda_t = np.random.normal(0,1)

        return mu + vol * lambda_t

class EarlyWarningIndicator(Data):
    # Maximum Likelihood Estimation
    def params(self, distribution: str, drop_extreme: float = None, window: int = 750, jump: int = 5) -> tuple:
        '''
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_extreme (float) : Drop sentiment index values above this threshold (optional)
            window (int) : Rolling window
            jump (int) : Date jump
        Return:
            tuple : MLE parameters
        '''
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)
        len_sentiment_index = len(sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy())

        likelihood = Likelihood(self.dict_data)
        params_list = []
        dates_list = []
        '''for multiplier in range(1, int(len_sentiment_index/window)+1):
            start = window * (multiplier - 1)
            end = window * multiplier
            params = likelihood.MLE(distribution, drop_extreme, start, end)'''

        a = datetime.now()
        for index, date in zip(range(window, len_sentiment_index + 1, jump), self.dates[sentiment_index_obj.L:][window::5]):
            start = index - window
            end = index
            params = likelihood.MLE(distribution, drop_extreme, start, end)
            params_list.append(params)
            dates_list.append(date)

        b = datetime.now()

        return params_list, (b-a)