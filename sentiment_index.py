import pandas as pd
import numpy as np
from data_preprocessor import Data
import matplotlib.pyplot as plt
from likelihood import Likelihood
from returns import Return

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