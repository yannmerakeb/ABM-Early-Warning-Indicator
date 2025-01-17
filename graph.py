import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessor import Data
# from sentiment_index import SentimentIndex
from returns import Return

class Graph(Data):

    def __init__(self, dict_data: dict):
        # Sentiment index
        super().__init__(dict_data)
        self.returns = Return(self.dict_data)
        # len_sentiment_index = len(sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy())
        self.EWI =

    def plot_index_prices(self):
        '''
        Plot the index prices
        '''
        '''if stock_name is not None:
            if detrended:
                return sentiment_index_obj.detrended_prices(stock_name)
            else:
                return self.stocks[stock_name]'''

        data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                          pd.DataFrame(self.index, columns=['Prices'])], axis=1)
        detrended_data = pd.concat([pd.DataFrame(self.dates, columns=['Date']),
                                    pd.DataFrame(self.returns.detrended_prices('index'), columns=['Detrended Prices'])], axis=1)

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


        '''stock_data = self.dict_data[index_name]['stocks'][stock_name]
        dates = self.dict_data[index_name]['dates']

        plt.figure(figsize=(12, 6))
        plt.plot(dates, stock_data, label=stock_name)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{stock_name} Stock Price')
        plt.legend()
        plt.show()'''

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
                                    pd.DataFrame(self.returns.detrended_prices(name),
                                                 columns=['Detrended Prices'])], axis=1)

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

    def plot_EWI(self):
