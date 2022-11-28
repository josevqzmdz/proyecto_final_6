import requests,json,numpy as np,pandas as pd
# https://towardsdatascience.com/using-recurrent-neural-networks-to-predict-bitcoin-btc-prices-c4ff70f9f3e4
## https://docs.coinranking.com/

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import time

from keras.models import load_model

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import cufflinks as cf

class BTC_predict:

    """
It accepts coin_id, timeframe, and currency parameters to clean the
historic coin data taken from COINRANKING.COM
It returns a Pandas Series with daily mean values of the
selected coin in which the date is set as the index
    """

    def history_price(self, coin_id=1335,timeframe = "5y",
                      coinID = "Qwsogvtv82FCd",
                 APIKey="YOUR-COINRANKING-API-KEY"):
        r = requests.get(
            "https://api.coinranking.com/v2/coin/" + coinID +
            "/history?timePeriod=" + timeframe + "&x-access-token=" + APIKey)
        coin = json.loads(r.text)['data']['history']  # Reading in json and cleaning the irrelevant parts
        df = pd.DataFrame(coin)

        df['price'] = pd.to_numeric(df['price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # .dt.time

        print(df.groupby('timestamp').mean()['price'])

        return df.groupby('timestamp').mean()['price']

    def price_matrix_creator(self, data, seq_len=30):
        '''
            It converts the series into a nested list where every item of the list contains historic prices of 30 days
        '''
        # https://www.statology.org/numpy-ndarray-object-has-no-attribute-append/
        #
        price_matrix = np.ndarray(shape=(100, 100))
        for index in range(len(data) - (seq_len+1)):
            price_matrix = np.stack(data[index:index+seq_len])
            #print(price_matrix.shape)
        return price_matrix

    def normarlize_windows(self, window_data):
        '''
            It normalizes each value to reflect the percentage changes
             from starting point
        '''
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window_data]
            normalised_data.append(normalised_window)
        return normalised_data

    def train_test_split(price_matrix, train_size=0.9, shuffle=False,
                         return_row=True):
        '''
        It makes a custom train test split where the last part is kept
         as the training set.
        '''
        price_matrix = np.array(price_matrix)#
        size = price_matrix.size
        kek = train_size * size
        #print(f"1): {kek}")
        row = len(kek)#
        #print(f"2): {row}")
        #print(price_matrix)
        train = price_matrix[:row, :]
        #train = price_matrix[:row, :-1]
        if shuffle == True:
            np.random.shuffle(train)
        X_train, y_train = train[:row, :-1], train[:row, -1]
        X_test, y_test = price_matrix[row:, :-1], price_matrix[row:, -1]#
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if return_row:
            return row, X_train, y_train, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test

    def deserializer(self, data, train_size=0.9, train_phase=False):
        '''
        Arguments:
        preds : Predictions to be converted back to their original values
        data : It takes the data into account because the normalization was made based on the full historic data
        train_size : Only applicable when used in train_phase
        train_phase : When a train-test split is made, this should be set to True so that a cut point (row) is calculated based on the train_size argument, otherwise cut point is set to 0

        Returns:
        A list of deserialized prediction values, original true values, and date values for plotting
        '''
        ser = self.history_price()
        price_matrix = np.array(self.price_matrix_creator(ser))
        if train_phase:
            row = int(round(train_size * len(price_matrix)))
        else:
            row = 0
        date = ser.index[row + 29:]
        date = np.reshape(date, (date.shape[0]))
        X_test = price_matrix[row:, :-1]
        y_test = price_matrix[row:, -1]
        preds_original = []
        preds = np.reshape(self, (self.shape[0]))
        for index in range(0, len(preds)):
            pred = (preds[index] + 1) * X_test[index][0]
            preds_original.append(pred)
        preds_original = np.array(preds_original)
        if train_phase:
            return [date, y_test, preds_original]
        else:
            import datetime
            return [date + datetime.timedelta(days=1), y_test]