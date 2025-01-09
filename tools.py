import numpy as np

class Return:
    def __init__(self, dict_data : dict):
        # Get the index and stock data
        self.index = dict_data['index']
        self.stocks = dict_data['stocks']

        # Get the names of the indexes and stocks
        self.index_names = self.index.columns[1]
        self.stock_names = list(self.stocks.keys())

    def cumulative_returns(self, index: bool):
        if index:
            ratio = (self.index.iloc[:,1] / self.index.iloc[0,1]).astype(float)
            return np.log(ratio)
        else:
            self.stock_cum_returns_dict = {}
            for stock in self.stock_names:
                ratio = (self.stocks[stock].iloc[:,1] / self.stocks[stock].iloc[0,1]).astype(float)
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
            return self.cumulative_returns(index).diff()[1:].reshape(-1,1)
        else:
            cum_returns = self.cumulative_returns(index=False)

            self.stock_returns_dict = {}
            for stock in self.stock_names:
                self.stock_returns_dict[stock] = cum_returns[stock].diff()[1:]
            return self.stock_returns_dict

    @property
    def avg_daily_return(self) -> np.array:
        '''
        OLS regression (cumulative_returns = cst + avg_daily_return * t => y = beta[0] + beta[1] * X)
        Return:
            np.array : beta
        '''
        cst_ones = np.ones((len(self.index[1:]), 1))
        self.t = self.index.index[1:].to_numpy().reshape(-1, 1)
        X = np.column_stack((cst_ones, self.t))
        y = self.cumulative_returns(index=True).to_numpy().reshape(-1,1)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta

    @property
    def detrended_returns(self) -> dict:
        '''
        Detrend stock returns
        Return:
            Detrended returns = returns * exp(-avg_daily_return * t)
        '''
        stock_returns = self.returns(index=False)

        self.detrended_stock_returns_dict = {}
        for stock in self.stock_names:
            self.detrended_stock_returns_dict[stock] = stock_returns[stock] * np.exp(-self.avg_daily_return[1] * self.t)
        return self.detrended_stock_returns_dict

'''class Graph:'''