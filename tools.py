import numpy as np

class Computation:
    def __init__(self, data):
        self.data = data

    @property
    def cumulative_returns(self):
        return np.log(self.data.iloc[:,1] / self.data.iloc[0,1])

    @property
    def returns(self):
        return self.cumulative_returns.diff()