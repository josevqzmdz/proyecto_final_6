from Neural_Network import *
from BTC_prediction.BTC_predict import BTC_predict
if __name__ == '__main__':
    # muestra las imagenes de numeros de mnist mas comunes
    e = Neural_Network()

    # llama las funciones de btc_predict
    btc = BTC_predict()
    history_price = btc.history_price()
    price_matrix = btc.price_matrix_creator(btc)
    price_matrix = btc.normarlize_windows(price_matrix)
    row, X_train, y_train, X_test, y_test = btc.train_test_split(price_matrix)


