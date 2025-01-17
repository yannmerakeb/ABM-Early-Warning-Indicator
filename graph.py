import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessor import Data
# from sentiment_index import SentimentIndex
from returns import Return
from tools import EarlyWarningIndicator


class Graph(Data):

    def __init__(self, dict_data: dict):
        # Sentiment index
        super().__init__(dict_data)
        self.prices = Return(self.dict_data)
        self.EWI = EarlyWarningIndicator(self.dict_data)

    def plot_index_prices(self):
        '''
        Plot the index prices
        '''
        data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                          pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                                    pd.DataFrame(self.prices.detrended_prices('index'), columns=['Detrended Prices'])], axis=1)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(data['Date'], data['Prices'])
        ax[0].set_title("S&P 500 Prices")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Prices")
        ax[1].plot(detrended_data['Date'], detrended_data['Detrended Prices'])
        ax[1].set_title("S&P 500 Detrended Prices")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Detrended Prices")
        plt.show()

    def plot_stock_prices(self, name: str):
        '''
        Plot the stock prices
        Args:
            name (str) : stock name
        Return:
            plt
        '''
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
        '''
        Plot the Early Warning Indicator
        Return:
            plt
        '''
        data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                          pd.DataFrame(self.EWI.estimation(0.95), columns=['EWI'])], axis=1)

        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['EWI'])
        plt.title("Early Warning Indicator")
        plt.xlabel("Date")
        plt.show()
