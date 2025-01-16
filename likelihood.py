import numpy as np
from data_preprocessor import Data
from scipy.optimize import minimize
from scipy.special import gammaln
import matplotlib.pyplot as plt
from scipy.stats import beta, chi2, norm
from sentiment_index import SentimentIndex

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