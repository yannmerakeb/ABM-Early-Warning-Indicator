import numpy as np
from data_preprocessor import Data

class Return(Data):

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