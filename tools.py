import pandas as pd
import numpy as np
from data_preprocessor import Data
from scipy.optimize import minimize
from scipy.special import gammaln

'''class Data:
    def __init__(self, dict_data: dict):
        # Get the index and stock data
        self.index = dict_data['index']
        self.stocks = dict_data['stocks']

        # Get the names of the indexes and stocks
        self.stock_names = list(self.stocks.keys())

        # Get the dates
        self.dates = dict_data['dates']'''

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
        cst_ones = np.ones((len(self.index[1:]), 1))
        t = np.arange(len(self.index[1:]))
        X = np.column_stack((cst_ones, t))
        y = self.cumulative_returns(index=True)[1:]
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta

    @property
    def detrended_returns(self) -> dict:
        '''
        Detrend stock returns
        Return:
            dict : Detrended returns = returns * exp(-avg_daily_return * t)
        '''
        # Number of days, including 0 for the first day (not detrended)
        t = np.arange(len(self.index))
        # t = self.index.index.to_numpy().reshape(-1, 1)
        detrend_factor = np.exp(-self.avg_daily_return[1] * t)

        self.detrended_stock_returns_dict = {}
        for stock in self.stock_names:
            self.detrended_stock_returns_dict[stock] = self.stocks[stock] * detrend_factor
        return self.detrended_stock_returns_dict

class SentimentIndex(Data):
    def __init__(self, dict_data: dict, EMA_window: int = 100):
        # Exponential Moving Average time window
        super().__init__(dict_data)
        self.L = EMA_window
        # Weight used for the EMA
        self.W = 2 / (self.L + 1)

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
            self.EMA_dict[stock] = ema_data'''

            # ema_data = pd.concat([self.stocks[stock]['Date'], pd.DataFrame(index=self.stocks[stock].index, columns=[stock])], axis=1)
            # ema_data.iloc[self.L] = np.mean(self.stocks[stock].iloc[:self.L, 1])

            ema_data = np.full(len(self.stocks[stock]), np.nan)
            ema_data[self.L] = np.mean(self.stocks[stock][:self.L])
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
                ema = self.W * self.stocks[stock][row] + (1 - self.W) * self.EMA_dict[stock][row - 1]
                self.EMA_dict[stock][row] = ema

            # Pessimistic state
            pessimistic_state = self.stocks[stock] < self.EMA_dict[stock]
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

class Modelling(Data):

    # Unconditional distribution of the sentiment index
    def beta_log_likelihood(self, params: tuple, sentiment_index: np.ndarray) -> float:
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

    def normal_log_likelihood(self, params: tuple, sentiment_index: np.ndarray) -> float:
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
        mu = sentiment_index + (e1 + e2) * (z_bar - sentiment_index) * b
        sigma = np.sqrt(2 * b * (1 - sentiment_index) * sentiment_index)

        # Log likelihood
        ll = (- 0.5 * np.sum(np.log(2 * np.pi * sigma[:-1] ** 2))
              - 0.5 * np.sum(((sentiment_index[1:] - mu[:-1]) / sigma[:-1]) ** 2))

        return -ll

    def MLE(self, distribution: str, drop_extreme: float = None) -> tuple:
        '''
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_extreme (float) : Drop sentiment index values above this threshold
        Return:
            tuple : MLE parameters
        '''
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)
        sentiment_index = sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy()

        # Drop extreme sentiment index values if specified
        if drop_extreme:
            sentiment_index = sentiment_index[sentiment_index <= drop_extreme]

        # Clip the sentiment index to avoid log(0) and log(1)
        epsilon = 1e-10
        sentiment_index = np.clip(sentiment_index, epsilon, 1 - epsilon)

        # Optimization options
        options = {'maxiter': 100000, 'ftol': 1e-10, 'gtol': 1e-10, 'xtol': 1e-10}

        # Maximum Likelihood Estimation
        if distribution.lower() == 'beta':
            params_init = np.array([1, 1])
            bounds = ((1e-10, None), (1e-10, None))
            MLE = minimize(self.beta_log_likelihood, params_init, args=(sentiment_index,),
                           bounds=bounds, method='L-BFGS-B', options=options)

        elif distribution.lower() == 'normal':
            params_init = np.array([1, 1, 1])
            bounds = ((1e-10, None), (1e-10, None), (1e-10, None))
            MLE = minimize(self.normal_log_likelihood, params_init, args=(sentiment_index,),
                           bounds=bounds, method='L-BFGS-B', options=options)

        else:
            raise ValueError("Distribution not supported")

        return MLE.x
