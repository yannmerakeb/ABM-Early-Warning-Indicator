import numpy as np
from typing_extensions import Optional
from data_preprocessor import Data
from likelihood import Likelihood

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