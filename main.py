from Neural_Network import *
from BTC_prediction.BTC_predict import BTC_predict as btc, BTC_predict

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

if __name__ == '__main__':
    # muestra las imagenes de numeros de mnist mas comunes
    #e = Neural_Network()

    # llama las funciones de btc_predict
    btcc = BTC_predict()
    history_price = btcc.history_price()
    price_matrix = btcc.price_matrix_creator(history_price)
    price_matrix = btcc.normarlize_windows(price_matrix)
    row, X_train, y_train, X_test, y_test = btcc.train_test_split(price_matrix)
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
    price_matrix = btc.price_matrix_creator(ser)
    X_test = btc.normalize_windows(price_matrix)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    """
    We will only have the normalized data for prediction: 
    No train-test split. We will also reshape the data manually to
     be able to use it in our saved model.

    After cleaning and preparing our data, we will load the trained 
    RNN model for prediction and predict tomorrow???s price.
    """
    model = load_model('coin_predictor.h5')
    preds = model.predict(X_test, batch_size=2)
    """
    However, our results will vary between -1 and 1, which will not make 
    a lot of sense. Therefore, we need to de-normalize them back to their 
    original values. We can achieve this with a custom function:
    """
    final_pred = btc.deserializer(preds, ser, train_size=0.9, train_phase=False)
    final_pred[1][0]

    """

    You may also be interested in the overall result of the RNN model
     and prefer to see it as a chart. We can also achieve these by
      using our X_test data from the training part of the tutorial.

    We will start by loading our model (consider this as an alternative
     to the single prediction case) and making the prediction on X_test
      data so that we can make predictions for a proper number of days
       for plotting with the following code:
    """
    model = load_model('coin_predictor.h5')
    preds = model.predict(X_test, batch_size=2)
    plotlist = btc.deserializer(preds, ser, train_phase=True)

    init_notebook_mode(connected=True)

    """
    After setting all the properties, we can finally plot our predictions 
    and observation values with the following code:
    """
    prices = pd.DataFrame({'Predictions': plotlist[1], 'Real Prices': plotlist[2]}, index=plotlist[0])
    iplot(prices.iplot(asFigure=True,
                       kind='scatter',
                       xTitle='Date',
                       yTitle='BTC Price',
                       title='BTC Price Predictions'))



