from data_preprocessor import Data
from datetime import datetime
from likelihood import Likelihood
from sentiment_index import SentimentIndex


class EarlyWarningIndicator(Data):
    # Maximum Likelihood Estimation
    def params(self, distribution: str, drop_extreme: float = None, window: int = 750, jump: int = 5) -> tuple:
        '''
        Compute the MLE
        Args:
            distribution (str) : Distribution to use for the MLE (Normal, Beta)
            drop_extreme (float) : Drop sentiment index values above this threshold (optional)
            window (int) : Rolling window
            jump (int) : Date jump
        Return:
            tuple : MLE parameters
        '''
        # Sentiment index
        sentiment_index_obj = SentimentIndex(self.dict_data)
        len_sentiment_index = len(sentiment_index_obj.sentiment_index.iloc[:, 1].to_numpy())

        likelihood = Likelihood(self.dict_data)
        params_list = []
        dates_list = []
        '''for multiplier in range(1, int(len_sentiment_index/window)+1):
            start = window * (multiplier - 1)
            end = window * multiplier
            params = likelihood.MLE(distribution, drop_extreme, start, end)'''

        a = datetime.now()
        for index, date in zip(range(window, len_sentiment_index + 1, jump), self.dates[sentiment_index_obj.L:][window::5]):
            start = index - window
            end = index
            params = likelihood.MLE(distribution, drop_extreme, start, end)
            params_list.append(params)
            dates_list.append(date)

        b = datetime.now()

        return params_list, (b-a)