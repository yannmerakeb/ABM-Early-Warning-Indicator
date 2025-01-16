import pandas as pd
import os
from tqdm import tqdm

class Data:
    def __init__(self, dict_data: dict):
        self.dict_data = dict_data

        # Get the index and stock data
        self.index = dict_data['index']
        self.stocks = dict_data['stocks']

        # Get the names of the indexes and stocks
        self.stock_names = list(self.stocks.keys())

        # Get the dates
        self.dates = dict_data['dates']

class DataPreprocessor:
    def __init__(self, directory_path):
        """
        Initialize the DataOrganizer with the directory path containing CSV files.
        Args:
            directory_path (str): Path to the folder containing the CSV files.
        """
        self.directory_path = directory_path

    def process_csv_files(self):
        """
        Process all CSV files in the directory to create DataFrames for each index.
        Return:
            dict: Two dictionaries of DataFrames, each corresponding to a specific index.
        """
        dataframes = {}

        # Get list of all CSV files in the specified directory
        for file_name in tqdm(os.listdir(self.directory_path), desc="Processing CSV files"):
            if file_name.endswith('.csv') and file_name == 'Nasdaq100.csv':
                file_path = os.path.join(self.directory_path, file_name)

                # Process each CSV file and extract data
                df = self._process_single_csv(file_path)

                '''# Format 'Date' column as datetime, and spot columns as float
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
                df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)'''

                # Store DataFrame by index name
                index_name = file_name.rsplit('.')[0]
                df_index = df.iloc[:,:2]


                stock_names = list(df.columns[2:])
                #df_stocks = {stock_name : df[['Date', stock_name]] for stock_name in stock_names}

                mat_index = df_index.iloc[:, 1].to_numpy()
                mat_stocks = {stock_name : df[stock_name].to_numpy() for stock_name in stock_names}

                dataframes[index_name] = {'index': mat_index, 'stocks': mat_stocks, 'dates':df['Date']}
                # dataframes[index_name] = {'index': df_index, 'stocks': df_stocks}

        return dataframes

    def _process_single_csv(self, file_path):
        """
        Process a single CSV file to extract the index and stock values.
        Args:
            file_path (str): Path to a CSV file.
        Return:
            pd.DataFrame: A DataFrame with the processed data."""
        # Read the raw CSV file, assuming ; is the delimiter
        raw_df = pd.read_csv(file_path, delimiter=';', header=None, low_memory=False)

        # Drop rows and columns with all missing values
        raw_df.dropna(axis=0, how='all', inplace=True)
        raw_df.dropna(axis=1, how='all', inplace=True)

        # Rename columns
        spots = raw_df.rename(columns = raw_df.iloc[0]).drop(raw_df.index[0])
        spots.columns = ['Date' if col % 2 == 0 else name for col, name in enumerate(spots.columns)]

        # Replace ',' by '.' before formating spot (non-date) columns as float
        spot_columns = [col for col in spots.columns if col != 'Date']
        spots.loc[:, spot_columns] = spots.loc[:, spot_columns].replace(',', '.', regex=True).astype(float)

        # Merge all spot columns on the 'Date' key
        spot = spots.iloc[:, :2]
        for col in range(2, len(spots.columns), 2):
            spot = spot.merge(spots.iloc[:, col:col + 2], on='Date', how='outer').dropna(how='all')

        # Drop rows with missing date/spot values
        spot.dropna(inplace=True)

        # Format 'Date' column as datetime
        spot['Date'] = pd.to_datetime(spot['Date'], format='%d/%m/%Y')

        # Sort DataFrame by 'Date', and reset index
        spot.sort_values(by='Date', inplace=True)
        spot = spot.reset_index(drop=True)

        return spot