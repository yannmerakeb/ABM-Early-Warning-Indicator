import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
from scipy.stats import beta
from datetime import datetime
from data_preprocessor import Data


class ReturnsAndPrices(Data):

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
        """
        Compute the returns of the index or the stocks
        Args:
            index (bool) : True if the index, False if the stocks
        Return:
            np.array or dict : returns
        """
        if index:
            return np.diff(self.cumulative_returns(index), axis=0).reshape(-1, 1)
        else:
            cum_returns = self.cumulative_returns(index=False)

            stock_returns_dict = {}
            for stock in self.stock_names:
                stock_returns_dict[stock] = cum_returns[stock].diff()[1:].to_numpy().reshape(-1,1)
            return stock_returns_dict

    def avg_daily_return(self):
        """
        OLS regression (cumulative_returns = cst + avg_daily_return * t => y = beta[0] + beta[1] * X)
        Return:
            np.array : beta
        """

        cst_ones = np.ones((len(self.index[1:]), 1))
        t = np.arange(1, len(self.index))

        X = np.column_stack((cst_ones, t))
        y = self.cumulative_returns(index=True)[1:]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta

    def detrended_prices(self, is_index: bool or str) -> dict:
        """
        Detrend stock prices
        Return:
            dict : Detrended prices = prices * exp(-avg_daily_return * t)
        """
        # Number of days, including 0 for the first day (not detrended)
        t = np.arange(len(self.index))
        # t = self.index.index.to_numpy().reshape(-1, 1)
        detrend_factor = np.exp(-self.avg_daily_return()[1] * t)

        if is_index:
            return self.index * detrend_factor
        else:
            self.detrended_prices_dict = {}
            for stock in self.stock_names:
                self.detrended_prices_dict[stock] = self.stocks[stock] * detrend_factor
            return self.detrended_prices_dict

    def formated_index_prices(self, window: int) -> tuple:
        """
        Format the index prices data
        Arg:
            shift (int) : shift the beginning date
        """
        dates = self.dates[window:]

        df = pd.concat([pd.DataFrame(dates, columns=['Date']),
                        pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_df = pd.concat([pd.DataFrame(dates, columns=['Date']),
                                  pd.DataFrame(self.detrended_prices(True), columns=['Detrended Prices'])],
                                 axis=1)

        return df, detrended_df

class SentimentIndex(Data):
    def __init__(self, dict_data: dict, EMA_window: int = 100):
        super().__init__(dict_data)
        # Exponential Moving Average time window
        self.L = EMA_window
        # Weight used for the EMA
        self.W = 2 / (self.L + 1)

        self.detrended_prices = ReturnsAndPrices(self.dict_data).detrended_prices(False)

    def _EMA_init(self) -> dict:
        """
        Initialize the EMA, the first value is the SMA
        Return:
            dict : initialized EMA
        """
        EMA_dict = {}

        for stock in self.stock_names:
            ema_data = np.full(len(self.detrended_prices[stock]), np.nan)
            ema_data[self.L] = np.mean(self.detrended_prices[stock][:self.L])
            EMA_dict[stock] = ema_data

        return EMA_dict

    def pessimistic_state(self) -> dict:
        """
        Compute the EMA by stock, and then the pessimistic state (binary)
        Return:
            dict : pessimistic state by stock
        """
        EMA_dict = self._EMA_init()
        pessimistic_state_dict = {}

        for stock in self.stock_names:
            for row in range(self.L + 1, len(self.stocks[stock])):
                # Exponential Moving Average
                ema = self.W * self.detrended_prices[stock][row] + (1 - self.W) * EMA_dict[stock][row - 1]
                EMA_dict[stock][row] = ema

            # Pessimistic state
            pessimistic_state = self.detrended_prices[stock] < EMA_dict[stock]
            pessimistic_state_dict[stock] = pessimistic_state

        return pessimistic_state_dict

    def sentiment_index(self) -> pd.DataFrame:
        """
        Compute the sentiment index and the daily changes
        Return:
            pd.DataFrame : sentiment index
        """
        pessimistic_states = self.pessimistic_state()
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
        """
        Compute the autocorrelation function of the sentiment index
        Return:
            plt
        """
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

    def theoretical_vs_empirical(self, distribution: str, drop_threshold: float = None):
        """
        Compare the theoretical distribution to the empirical distribution
        Args:
            distribution (str): Theoretical distribution to compare ('beta' or 'normal')
            drop_threshold (float): Drop sentiment index values above this threshold (optional)
        """
        # Sentiment index
        sentiment_index = SentimentIndex(self.dict_data).sentiment_index().iloc[:, 1].to_numpy()

        # Parameters
        x = np.linspace(0.05, 0.95, 100)
        if distribution == 'beta':
            e1, e2 = self.MLE('beta')
            theoretical_distribution = beta.pdf(x, e1, e2)
        elif distribution == 'normal':
            mc = MonteCarlo(self.dict_data)
            simulated_sentiment_index = mc.simulation(drop_threshold)
            theoretical_distribution = simulated_sentiment_index.flatten()
        else:
            raise ValueError("Distribution not supported")

        # Calculate the histogram data
        counts, bins = np.histogram(sentiment_index, bins=30, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plot the points at the tops of the histogram bars
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_centers, counts, color='blue', label='Empirical')
        if distribution == 'normal':
            sns.kdeplot(theoretical_distribution, color='red', label='Theoretical')
        else:
            plt.plot(x, theoretical_distribution, color='red', label='Theoretical')
        plt.xlabel('Sentiment Index')
        plt.ylabel('Density')
        plt.title(f'Theoretical vs Empirical Distribution of the Sentiment Index ({distribution.capitalize()})')
        plt.legend()
        plt.show()

    # Unconditional distribution of the sentiment index
    def neg_beta_log_likelihood(self, params: tuple, sentiment_index: np.ndarray) -> float:
        """
        Compute the beta log likelihood
        Return:
            float : beta log likelihood
        """
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
        """
        Compute the Normal log likelihood
        Args:
            params (tuple) : Normal parameters
            sentiment_index (np.ndarray) : Sentiment index
        Return:
            float : Normal log likelihood
        """
        # Parameters
        e1, e2, b = params

        # Mean and standard deviation
        z_bar = e1 / (e1 + e2)
        mu = (sentiment_index + (e1 + e2) * (z_bar - sentiment_index) * b)[:-1]
        sigma = np.sqrt(2 * b * (1 - sentiment_index) * sentiment_index)[:-1]

        # Log likelihood
        ll = (- 0.5 * np.sum(np.log(2 * np.pi * sigma ** 2))
              - 0.5 * np.sum(((sentiment_index[1:] - mu) / sigma) ** 2))

        return -ll

    # Maximum Likelihood Estimation
    def MLE(self, distribution: str, drop_threshold: float = None, start: int = None, end: int = None) -> tuple:
        """
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_threshold (float) : Drop sentiment index values above this threshold (optional)
            start (int) : Start date index
            end (int) : End date index
        Return:
            tuple : MLE parameters
        """
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)

        if start is None or end is None:
            sentiment_index = sentiment_index_obj.sentiment_index().iloc[:, 1].to_numpy()
        else:
            sentiment_index = sentiment_index_obj.sentiment_index().iloc[:, 1].to_numpy()[start:end]

        # Drop extreme sentiment index values if specified
        if drop_threshold:
            sentiment_index = sentiment_index[sentiment_index <= drop_threshold]

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

            return MLE.x

        else:
            raise ValueError("Distribution not supported")

class MonteCarlo(Data):

    def __init__(self, dict_data, num_simulations: int = 1000, num_days: int = 1000):
        """
        Initialize the Monte Carlo simulation
        Args:
            num_simulations (int) : Number of simulations to run
            num_days (int) : Number of days to simulate
        """
        super().__init__(dict_data)
        self.num_simulations = num_simulations
        self.num_days = num_days

    def simulation(self, drop_threshold: float = None) -> np.ndarray:
        """
        Perform Monte Carlo simulation based on the Normal distribution and parameters
        Args:
            drop_threshold (float) : Drop sentiment index values above this threshold (optional)
        Return:
            np.ndarray : Simulated paths
        """
        likelihood = Likelihood(self.dict_data)

        # Parameters
        e1, e2, b = likelihood.MLE('normal', drop_threshold)

        # Initialize the simulated sentiment index with zeros
        simulated_sentiment_index = np.zeros((self.num_days, self.num_simulations))

        # Simulate the sentiment index
        for simul in range(self.num_simulations):
            for day in range(1, self.num_days):
                while not (0 < simulated_sentiment_index[day, simul] < 1):
                    simulated_sentiment_index[day, simul] = self._individual_simulation(simulated_sentiment_index[day - 1, simul],
                                                                                        e1, e2, b)

        return simulated_sentiment_index

    def _individual_simulation(self, z_t, e1, e2, b) -> float:
        """
        Compute the next sentiment index value based on the Normal distribution
        Args:
            z_t (float) : Current sentiment index value
            e1, e2, b (float) : Parameters used to compute the mean and standard deviation of the Normal distribution
        """
        # Compute the mean and standard deviation of the Normal distribution at each time step
        z_bar = e1 / (e1 + e2)
        mu = z_t + (e1 + e2) * (z_bar - z_t) * b
        vol = np.sqrt(2 * b * (1 - z_t) * z_t)

        # Generate a random Brownian increment with size 1
        lambda_t = np.random.normal(0,1)

        return mu + vol * lambda_t

class EarlyWarningIndicator(Data):

    def __init__(self, dict_data: dict, distribution: str = 'beta', drop_threshold: float = 0.95, window: int = 750, jump: int = 25):
        """
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_threshold (float) : Drop sentiment index values above this threshold (optional)
            window (int) : Rolling window
            jump (int) : Date jump
        """
        super().__init__(dict_data)
        self.distribution = distribution
        self.drop_threshold = drop_threshold
        self.window = window
        self.jump = jump

    def estimation(self) -> tuple:
        """
        Estimate the MLE
        Return:
            tuple : MLE parameters
        """
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)
        len_sentiment_index = len(sentiment_index_obj.sentiment_index().iloc[:, 1].to_numpy())

        likelihood = Likelihood(self.dict_data)
        params_list = []
        dates_list = []
        percentiles_90 = []
        percentiles_10 = []


        a = datetime.now()
        for index, date in zip(range(self.window, len_sentiment_index + 1, self.jump), self.dates[sentiment_index_obj.L:][self.window::self.jump]):
            start = index - self.window
            end = index
            params = likelihood.MLE(self.distribution, self.drop_threshold, start, end)
            params_list.append(params[1] - params[0])
            dates_list.append(date)
            percentiles_90.append(np.percentile(params_list, 90))
            percentiles_10.append(np.percentile(params_list, 10))

        b = datetime.now()

        return dates_list, params_list, percentiles_90, percentiles_10

    def formated_EWI(self):
        """
        Format the Early Warning Indicator data
        """
        # EWI = EarlyWarningIndicator(self.dict_data, 'beta', 0.95, 750, 25)
        dates, params, percentils_90, percentils_10 = self.estimation()

        # Create a DataFrame with the dates and the Early Warning Indicator
        df = pd.concat([pd.DataFrame(dates, columns=['Date']), pd.DataFrame(params, columns=['EWI'])], axis=1)

        return df, self.window, percentils_90, percentils_10, dates

class TradingStrategies(Data):
    def __init__(self, dict_data: dict, drop_threshold: float = None, window: int = 750, jump: int = 75,
                 initial_capital: float = 1000):
        """
        Initialize the trading strategy class
        Args:
            dict_data (dict) : Data dictionary
            drop_threshold (float) : Drop sentiment index values above this threshold
            window (int) : Rolling window used for the parameter estimation
            jump (int) : Date jump
        """
        super().__init__(dict_data)

        # Early Warning Indicator
        self.EWI = EarlyWarningIndicator(self.dict_data, 'beta', drop_threshold, window, jump)
        self.df_EWI, self.window, self.percentils_90, self.percentils_10, self.dates = self.EWI.formated_EWI()

        # Index Prices
        self.df_index = ReturnsAndPrices(self.dict_data).formated_index_prices(self.EWI.window)

        # Initial capital
        self.initial_capital = initial_capital

    def signals(self):
        """
        Compute the trading signals
        """
        signals = np.where(self.df_EWI['EWI'] > self.percentils_90, 1,
                           np.where(self.df_EWI['EWI'] < self.percentils_10, -1, 0))

        return signals

    def strategy(self):
        """
        Compute the trading strategy
        """
        signals = self.signals()



        df_prices, df_detrended_prices = self.returns_and_prices.formated_index_prices(window)


class Graph(Data):

    def __init__(self, dict_data: dict, drop_threshold: float, window: int = 750, jump: int = 5):
        super().__init__(dict_data)
        self.returns_and_prices = ReturnsAndPrices(self.dict_data)
        self.EWI = EarlyWarningIndicator(self.dict_data, 'beta', drop_threshold, window, jump)

    def plot_prices_and_EWI(self):
        """
        Plot the prices (trended and detrended) and the Early Warning Indicator
        """
        df_EWI, window, percentils_90, percentils_10, dates = self.EWI.formated_EWI()
        df_prices, df_detrended_prices = self.returns_and_prices.formated_index_prices(window)

        fig, ax = plt.subplots(3, 1, figsize=(21, 14))

        # Plot the Early Warning Indicator
        ax[0].plot(df_EWI['Date'], df_EWI['EWI'])
        ax[0].plot(df_EWI['Date'], percentils_90, color='green', linestyle='--', label='90th percentile (bull market)')
        ax[0].plot(df_EWI['Date'], percentils_10, color='red', linestyle='--', label='10th percentile (bear market)')
        ax[0].fill_between(df_EWI['Date'], df_EWI['EWI'].min(), df_EWI['EWI'].max(), where=(df_EWI['EWI'] > percentils_90), color='green', alpha=0.3)
        ax[0].fill_between(df_EWI['Date'], df_EWI['EWI'].min(), df_EWI['EWI'].max(), where=(df_EWI['EWI'] < percentils_10), color='red', alpha=0.3)
        ax[0].set_title(f"{self.index_name} Early Warning Indicator")
        ax[0].set_xlabel("Date")

        # Plot the prices
        ax[1].plot(df_prices['Date'], df_prices['Prices'])
        ax[1].fill_between(df_EWI['Date'], df_prices['Prices'].min(), df_prices['Prices'].max(), where=(df_EWI['EWI'] > percentils_90), color='green', alpha=0.3)
        ax[1].fill_between(df_EWI['Date'], df_prices['Prices'].min(), df_prices['Prices'].max(), where=(df_EWI['EWI'] < percentils_10), color='red', alpha=0.3)
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Prices")
        ax[1].set_title(f"{self.index_name} Prices")

        # Plot the detrended prices
        ax[2].plot(df_detrended_prices['Date'], df_detrended_prices['Detrended Prices'])
        ax[2].fill_between(df_EWI['Date'], df_detrended_prices['Detrended Prices'].min(), df_detrended_prices['Detrended Prices'].max(), where=(df_EWI['EWI'] > percentils_90), color='green', alpha=0.3)
        ax[2].fill_between(df_EWI['Date'], df_detrended_prices['Detrended Prices'].min(), df_detrended_prices['Detrended Prices'].max(), where=(df_EWI['EWI'] < percentils_10), color='red', alpha=0.3)
        ax[2].set_xlabel("Date")
        ax[2].set_ylabel("Detrended Prices")
        ax[2].set_title(f"{self.index_name} Detrended Prices")

        plt.tight_layout()
        plt.show()

    '''def formated_index_prices(self, window: int):
        """
        Format the index prices data
        Arg:
            shift (int) : shift the beginning date
        """
        dates = self.dates[window:]

        df = pd.concat([pd.DataFrame(dates, columns=['Date']),
                          pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_df = pd.concat([pd.DataFrame(dates, columns=['Date']),
                                    pd.DataFrame(self.prices.detrended_prices(True), columns=['Detrended Prices'])],
                                   axis=1)
        
        return df, detrended_df'''

    '''def formated_index_prices(self, shift: int = 750):
        """
        Format the index prices data
        Arg:
            shift (int) : shift the beginning date
        """
        # dates = self.dates[shift:]

        data = pd.concat([pd.DataFrame(dates, columns=['Date']),
                          pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_data = pd.concat([pd.DataFrame(dates, columns=['Date']),
                                    pd.DataFrame(self.prices.detrended_prices(True), columns=['Detrended Prices'])], axis=1)

        

        fig, ax = plt.subplots(3, 1, figsize=(18, 8))
        ax[0].plot(data['Date'], data['Prices'])
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Prices")
        ax[0].set_title(f"{self.index_name} Prices")
        ax[1].plot(detrended_data['Date'], detrended_data['Detrended Prices'])
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Detrended Prices")
        ax[1].set_title(f"{self.index_name} Detrended Prices")

        ax[2].plot(data1['Date'], data1['EWI'])
        ax[2].plot(data1['Date'], percentils_90, color='green', linestyle='--', label='90th percentile (bull market)')
        ax[2].plot(data1['Date'], percentils_10, color='red', linestyle='--', label='10th percentile (bear market)')
        ax[2].set_title("Early Warning Indicator")
        ax[2].set_xlabel("Date")

        plt.show()'''

    '''def formated_EWI(self):
        """
        Format the Early Warning Indicator data
        """
        EWI = EarlyWarningIndicator(self.dict_data, 'beta', 0.95, 750, 25)
        dates, params, percentils_90, percentils_10 = EWI.estimation

        # Create a DataFrame with the dates and the Early Warning Indicator
        df = pd.concat([pd.DataFrame(dates, columns=['Date']), pd.DataFrame(params, columns=['EWI'])], axis=1)
        
        return df, EWI.window, percentils_90, percentils_10'''

    '''def plot_index_prices(self, shift: int = 750):
        """
        Plot the index prices
        Arg:
            shift (int) : shift the beginning date
        """
        dates = self.dates[shift:]

        data = pd.concat([pd.DataFrame(dates, columns=['Date']),
                          pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_data = pd.concat([pd.DataFrame(dates, columns=['Date']),
                                    pd.DataFrame(self.prices.detrended_prices(True), columns=['Detrended Prices'])], axis=1)

        EWI = EarlyWarningIndicator(self.dict_data, 'beta', 0.95, 750, 25)
        dates, params, percentils_90, percentils_10 = EWI.estimation

        # Create a DataFrame with the dates and the Early Warning Indicator
        data1 = pd.concat([pd.DataFrame(dates, columns=['Date']), pd.DataFrame(params, columns=['EWI'])], axis=1)

        fig, ax = plt.subplots(3, 1, figsize=(18, 8))
        ax[0].plot(data['Date'], data['Prices'])
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Prices")
        ax[0].set_title(f"{self.index_name} Prices")
        ax[1].plot(detrended_data['Date'], detrended_data['Detrended Prices'])
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Detrended Prices")
        ax[1].set_title(f"{self.index_name} Detrended Prices")

        ax[2].plot(data1['Date'], data1['EWI'])
        ax[2].plot(data1['Date'], percentils_90, color='green', linestyle='--', label='90th percentile (bull market)')
        ax[2].plot(data1['Date'], percentils_10, color='red', linestyle='--', label='10th percentile (bear market)')
        ax[2].set_title("Early Warning Indicator")
        ax[2].set_xlabel("Date")

        plt.show()

    def plot_stock_prices(self, name: str):
        """
        Plot the stock prices
        Args:
            name (str) : stock name
        Return:
            plt
        """
        data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                          pd.DataFrame(self.stocks[name], columns=['Prices'])], axis=1)
        detrended_data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                                    pd.DataFrame(self.prices.detrended_prices(name),
                                                 columns=['Detrended Prices'])], axis=1)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(data['Date'], data['Prices'])
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Prices")
        ax[1].plot(detrended_data['Date'], detrended_data['Detrended Prices'])
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Detrended Prices")
        plt.show()

    def plot_EWI(self):
        """
        Plot the Early Warning Indicator
        Return:
            plt
        """
        EWI = EarlyWarningIndicator(self.dict_data, 'beta', 0.95, 750, 25)
        dates, params, percentils_90, percentils_10 = EWI.estimation

        # Create a DataFrame with the dates and the Early Warning Indicator
        data = pd.concat([pd.DataFrame(dates, columns=['Date']), pd.DataFrame(params, columns=['EWI'])], axis=1)

        # Plot the Early Warning Indicator
        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['EWI'])
        plt.axhline(y=percentils_90, color='darkgray', linestyle='--', label='90th percentile (bull market)')
        plt.axhline(y=percentils_10, color='lightgray', linestyle='--', label='10th percentile (bear market)')
        plt.title("Early Warning Indicator")
        plt.xlabel("Date")

        plt.tight_layout()
        plt.show()'''
