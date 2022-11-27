import requests,json,numpy as np,pandas as pd
# https://towardsdatascience.com/using-recurrent-neural-networks-to-predict-bitcoin-btc-prices-c4ff70f9f3e4
## https://docs.coinranking.com/

class BTC_predict:
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
            It normalizes each value to reflect the percentage changes from starting point
        '''

