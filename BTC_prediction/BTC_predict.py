import requests,json,numpy as np,pandas as pd
# https://towardsdatascience.com/using-recurrent-neural-networks-to-predict-bitcoin-btc-prices-c4ff70f9f3e4
## https://docs.coinranking.com/

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import time

from keras.models import load_model

class BTC_predict:

    def __init__(self):
        ser = BTC_predict()
        history_price = btc.history_price()
        price_matrix = btc.price_matrix_creator(btc)
        price_matrix = btc.normarlize_windows(price_matrix)
        row, X_train, y_train, X_test, y_test = btc.train_test_split(price_matrix)
        """
        After preparing our data, it is time for building the model
        that we will later train by using the cleaned&normalized data.
        We will start by importing our Keras components and setting some
         parameters with the following code:
        """
        # LSTM Model parameters, I chose
        batch_size = 2  # Batch size (you may try different values)
        epochs = 15  # Epoch (you may try different values)
        seq_len = 30  # 30 sequence data (Representing the last 30 days)
        loss = 'mean_squared_error'  # Since the metric is MSE/RMSE
        optimizer = 'rmsprop'  # Recommended optimizer for RNN
        activation = 'linear'  # Linear activation
        input_shape = (None, 1)  # Input dimension
        output_dim = 30  # Output dimension
        model = Sequential()
        model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
        model.add(Dense(units=32, activation=activation))
        model.add(LSTM(units=output_dim, return_sequences=False))
        model.add(Dense(units=1, activation=activation))
        model.compile(optimizer=optimizer, loss=loss)
        """
        Now it is time to train our model with the cleaned data. 
        You can also measure the time spent during the training. 
        Follow these codes:
        """
        start_time = time.time()
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.05)
        end_time = time.time()
        processing_time = end_time - start_time
        # dont forget to save it
        model.save('coin_predictor.h5')
        """
        After we train the model, we need to obtain the current data 
        for predictions, and since we normalize our data, predictions 
        will also be normalized. Therefore, we need to de-normalize
         back to their original values. Firstly, we will obtain the 
         data with a similar, partially different, manner with the 
         following code:
        """
        # We need ser, preds, row
        ser = history_price(timeframe='30d')[1:31]
        price_matrix = self.price_matrix_creator(ser)
        X_test = self.normalize_windows(price_matrix)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        """
        We will only have the normalized data for prediction: 
        No train-test split. We will also reshape the data manually to
         be able to use it in our saved model.

        After cleaning and preparing our data, we will load the trained 
        RNN model for prediction and predict tomorrow’s price.
        """
        model = load_model('coin_predictor.h5')
        preds = model.predict(X_test, batch_size=2)
        """
        However, our results will vary between -1 and 1, which will not make 
        a lot of sense. Therefore, we need to de-normalize them back to their 
        original values. We can achieve this with a custom function:
        """
        final_pred = self.deserializer(preds, ser, train_size=0.9, train_phase=False)
        final_pred[1][0]

    '''
    It accepts coin_id, timeframe, and currency parameters to clean the
    historic coin data taken from COINRANKING.COM
        It returns a Pandas Series with daily mean values of the
        selected coin in which the date is set as the index
        '''
    def history_price(self, coin_id=1335,timeframe = "5y",coinID = "Qwsogvtv82FCd",
                 APIKey="YOUR-COINRANKING-API-KEY"):
        r = requests.get(
            "https://api.coinranking.com/v2/coin/" + coinID + "/history?timePeriod=" + timeframe + "&x-access-token=" + APIKey)
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
        price_matrix = []
        for index in range(len(data) - (seq_len+1)):
            price_matrix.append(data[index:index+seq_len])
        return price_matrix

    def normarlize_windows(self, window_data):
        '''
            It normalizes each value to reflect the percentage changes
             from starting point
        '''
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data

    def train_test_split_(price_matrix, train_size=0.9, shuffle=False,
                          return_row=True):
        '''
        It makes a custom train test split where the last part is kept
         as the training set.
        '''
        price_matrix = np.array(price_matrix)
        # print(price_matrix.shape)
        row = int(round(train_size * len(price_matrix)))
        train = price_matrix[:row, :]
        if shuffle == True:
            np.random.shuffle(train)
        X_train, y_train = train[:row, :-1], train[:row, -1]
        X_test, y_test = price_matrix[row:, :-1], price_matrix[row:, -1]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if return_row:
            return row, X_train, y_train, X_test, y_test
        else:
            X_train, y_train, X_test, y_test

    def deserializer(self, data, train_size=0.9, ser, train_phase=False):
        '''
        Arguments:
        preds : Predictions to be converted back to their original values
        data : It takes the data into account because the normalization was made based on the full historic data
        train_size : Only applicable when used in train_phase
        train_phase : When a train-test split is made, this should be set to True so that a cut point (row) is calculated based on the train_size argument, otherwise cut point is set to 0

        Returns:
        A list of deserialized prediction values, original true values, and date values for plotting
        '''
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