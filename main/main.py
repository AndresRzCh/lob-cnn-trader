import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from binance.client import Client
from binance.websockets import BinanceSocketManager
from sklearn.preprocessing import StandardScaler
from binance.depthcache import DepthCacheManager
from datetime import date
import time
from binance.exceptions import BinanceAPIException
import subprocess

# Settings
api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')
client = Client(api_key, secret_key)
bm = BinanceSocketManager(client)
subprocess.getoutput('net start w32time')

class Trader:

    def __init__(self, assets, model_date, treshold, cache='../data/cache.pickle', stats='../data/stats.pickle',
                 reset_stats=False, reset_cache=False, demo=0, limit_depth=50, window=100, limit_file=3600, leverage=1, fee=0.00036,
                 size=1, confirm=1, max_in_count=10, max_out_count=10, expiry_position=300, expiry_order=10, trailing=0.0001, sl=0.01, tp=0.001):

        self.demo = demo == 0
        self.stats = list() if reset_stats else pickle.load(open(stats, 'rb'))
        self.cache = dict() if reset_cache else pickle.load(open(cache, 'rb'))

        try:
            self.free_balance = float(client.futures_account_balance()[0]['balance'])
        except BinanceAPIException:
            print(subprocess.getoutput('w32tm /resync'))
            self.free_balance = float(client.futures_account_balance()[0]['balance'])

        self.trailing = trailing
        self.tp = tp
        self.sl = sl

        self.fee = fee
        self.info = pd.DataFrame(client.futures_exchange_info()['symbols']).set_index('symbol')
        self.model = {asset: load_model(f'../data/{asset}/{model_date}/model.h5') for asset in assets}
        self.data = {asset: list() for asset in assets}
        self.buffer = {asset: list() for asset in assets}
        self.limit = limit_depth
        self.limit_file = limit_file
        self.counter = {asset: 0 for asset in assets}
        self.scalers = {asset: None for asset in assets}
        self.confirm_long = {asset: 0 for asset in assets}
        self.confirm_short = {asset: 0 for asset in assets}
        self.in_counter = {asset: 0 for asset in assets}
        self.out_counter = {asset: 0 for asset in assets}
        self.tp_price = {asset: 0 for asset in assets}
        self.sl_price = {asset: 0 for asset in assets}
        self.max_in_count = max_in_count
        self.max_out_count = max_out_count
        self.prices = {asset: {'ask': 0, 'bid': 0} for asset in assets}
        self.assets = assets
        self.window = window
        self.leverage = leverage
        self.df = pd.DataFrame()
        self.expiry_position = expiry_position
        self.expiry_order = expiry_order
        self.confirm = confirm
        self.mid_price = 0
        self.size = size
        self.demoId = 0
        self.treshold = treshold
        print(f'Balance Updated: {self.free_balance:.2f} $')

        for asset in assets:
            if asset not in self.cache.keys():
                self.cache[asset] = {'type': None, 'direction': None, 'order': {'status': 'NONE'}}

        for asset in list(self.cache.keys()):
            if asset not in assets:
                self.cache.pop(asset)

        pickle.dump(self.stats, open(stats, 'wb'))
        pickle.dump(self.cache, open(cache, 'wb'))

    def affordable(self, asset):
        if self.cache[asset]['type'] == 'LONG':
            price = self.prices[asset]['bid']
        elif self.cache[asset]['type'] == 'SHORT':
            price = self.prices[asset]['ask']
        else:
            price = self.free_balance

        minqty = float(self.info.loc[asset, 'filters'][1]['minQty'])
        qty = np.floor((self.leverage * self.size * (self.free_balance / price) * (1 / len(self.assets))) * (1 / minqty)) * minqty

        print(f'Checking if affordable. Quantity: {qty}. Min Quantity: {minqty}')

        if qty >= minqty:
            return {'affordable': True, 'qty': str(qty), 'price': str(price)}
        else:
            return {'affordable': False, 'qty': '0', 'price': '0'}

    def filling_download(self, asset):
        return self.counter[asset] < self.limit_file

    def save_download(self, asset):
        self.counter[asset] = 0
        path = f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = f'{path}/LOB{str(len(list(filter(lambda k: "LOB" in k, os.listdir(path))))).zfill(3)}.pickle'
        pickle.dump(self.data[asset], open(filename, 'wb'))
        print(f'Saved Downloaded Data at: {filename}')
        self.data[asset] = list()
        return None

    def filling_buffer(self, asset):
        return len(self.buffer[asset]) < self.window

    def fill_buffer(self, asset, depth_cache):
        print(f'Filling Buffer: {len(self.buffer[asset])+1}/{self.window}')
        self.buffer[asset].append({'t': depth_cache.update_time, 'a': depth_cache.get_asks()[:self.limit], 'b': depth_cache.get_bids()[:self.limit]})
        return None

    def missing_scalers(self, asset):
        return self.scalers[asset] is None

    def fill_scalers(self, asset):
        if os.path.isfile(f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}/scalers.pickle'):
            self.scalers[asset] = pickle.load(open(f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}/scalers.pickle', 'rb'))
            print('Scalers Loaded')
        else:
            if not os.path.exists(f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}'):
                os.makedirs(f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}')
            df_unscaled = pd.DataFrame([np.insert(np.array(f['a'][:self.limit][::-1] + f['b'][:self.limit]).flatten('F'), 0, f['t']) for f in self.buffer[asset]])
            df_unscaled = df_unscaled.set_index(0)
            self.scalers[asset] = [StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()]
            for scaler, i in zip(self.scalers[asset], range(4)):
                scaler.fit(df_unscaled.iloc[:, i * self.limit:(i + 1) * self.limit].values)
            self.buffer[asset] = list()
            pickle.dump(self.scalers[asset], open(f'../data/{asset}/{date.today().strftime("%Y_%m_%d")}/scalers.pickle', 'wb'))
            print('Scalers Created')

    def update_buffer(self, asset, depth_cache):
        self.buffer[asset].append({'t': depth_cache.update_time, 'a': depth_cache.get_asks()[:self.limit], 'b': depth_cache.get_bids()[:self.limit]})
        self.buffer[asset].pop(0)
        self.df = pd.DataFrame([np.insert(np.array(f['a'][:self.limit][::-1] + f['b'][:self.limit]).flatten('F'), 0, f['t']) for f in self.buffer[asset]]).set_index(0)
        aux = client.futures_orderbook_ticker(symbol=asset)
        self.prices[asset] = {'ask': float(aux['askPrice']), 'bid': float(aux['bidPrice'])}
        self.mid_price = (self.prices[asset]['ask'] + self.prices[asset]['bid']) / 2

    def confirmed_position(self, asset):
        return self.cache[asset]['order']['status'] == 'FILLED'

    def pending_position(self, asset):

        if self.demo:
            self.cache[asset]['order']['status'] = 'FILLED'
        else:
            self.cache[asset]['order'] = client.futures_get_order(symbol=asset, orderId=self.cache[asset]['order']['orderId'])

        print(f'Updated Position Status: {self.cache[asset]["order"]["status"]}')

        if self.cache[asset]['direction'] == 'OUT' and self.cache[asset]['order']['status'] == 'FILLED':
            if self.demo:
                if self.cache[asset]['type'] == 'LONG':
                    pnl = (float(self.mid_price) / float(self.cache[asset]["order"]["price"]) - 1)
                else:
                    pnl = (1 - float(self.mid_price) / float(self.cache[asset]["order"]["price"]))
                self.free_balance *= (1 + pnl - self.fee)
                print(f'Balance Updated: {self.free_balance:.2f} $')
            else:
                self.free_balance = float(client.futures_account_balance()[0]['balance'])
                print(f'Balance Updated: {self.free_balance:.2f} $')
            self.cache[asset] = {'type': None, 'direction': None, 'order': {'status': 'NONE'}}
            return False

        return self.cache[asset]['order']['status'] == 'NEW'

    def no_position(self, asset):
        return self.cache[asset]['order']['status'] == 'NONE'

    def position_expiry(self, asset):
        time_elapsed = int(time.time() * 1000) - self.cache[asset]['order']['updateTime']
        return time_elapsed >= self.expiry_position * 1000

    def hit_sl(self, asset):
        if self.cache[asset]['type'] == 'LONG':
            return self.mid_price <= self.sl_price[asset]
        return self.mid_price >= self.sl_price[asset]

    def hit_tp(self, asset):
        if self.cache[asset]['type'] == 'LONG':
            return self.mid_price >= self.tp_price[asset]
        return self.mid_price <= self.tp_price[asset]

    def update_trailing(self, asset):
        if self.cache[asset]['type'] == 'LONG':
            self.tp_price[asset] = self.mid_price * (1 + self.trailing)
            self.sl_price[asset] = self.mid_price * (1 - self.trailing)
        else:
            self.tp_price[asset] = self.mid_price * (1 - self.trailing)
            self.sl_price[asset] = self.mid_price * (1 + self.trailing)
        print('Trailing Updated')

    def order_expiry(self, asset):
        time_elapsed = int(time.time() * 1000) - self.cache[asset]['order']['updateTime']
        return time_elapsed >= self.expiry_order * 1000

    def signal(self, asset):

        for scaler, i in zip(self.scalers[asset], range(4)):
            self.df.iloc[:, i * self.limit:(i + 1) * self.limit] = scaler.transform(self.df.iloc[:, i * self.limit:(i + 1) * self.limit])
            prediction = self.model[asset].predict(self.df.values.reshape((1, self.window, self.limit * 4, 1)))

        if np.max(prediction) > self.treshold:
            signal = np.argmax(prediction)
            if signal == 0:
                self.cache[asset]['type'] = 'LONG'
                print(f'Signal: {self.cache[asset]["type"]}. Values: {np.around(prediction, 2)}')
                return True
            elif signal == 1:
                self.cache[asset]['type'] = 'SHORT'
                print(f'Signal: {self.cache[asset]["type"]}. Values: {np.around(prediction, 2)}')
                return True

        self.cache[asset]['type'] = None
        # print(f'Signal: {self.cache[asset]["type"]}. Values: {np.around(prediction, 2)}')
        self.confirm_short[asset] = 0
        self.confirm_long[asset] = 0
        return False

    def long_confirmed(self, asset):
        if self.confirm_long[asset] >= self.confirm:
            self.confirm_long[asset] = 0
            print('Long Confirmed')
            return True
        self.confirm_long[asset] += 1
        self.confirm_short[asset] = 0
        return False

    def short_confirmed(self, asset):
        if self.confirm_short[asset] >= self.confirm:
            self.confirm_short[asset] = 0
            print('Short Confirmed')
            return True
        self.confirm_short[asset] += 1
        self.confirm_long[asset] = 0
        return False

    def open_position(self, asset):

        if self.cache[asset]['type'] == 'LONG' and self.long_confirmed(asset):

            aux = self.affordable(asset)
            if aux['affordable']:

                if self.demo:
                    order = {'orderId': self.demoId, 'status': 'NEW', 'price': aux['price'], 'executedQty': aux['qty'], 'updateTime': int(time.time() * 1000)}
                    self.demoId += 1
                else:
                    order = client.futures_create_order(symbol=asset, side='BUY', type='LIMIT', price=aux['price'], quantity=aux['qty'], timeInForce='GTC')

                self.cache[asset]['type'] = 'LONG'
                self.cache[asset]['direction'] = 'IN'
                self.tp_price[asset] = self.mid_price * (1 + self.tp)
                self.sl_price[asset] = self.mid_price * (1 - self.sl)
                self.cache[asset]['order'] = order
                print(f'Buy order sent at {self.cache[asset]["order"]["price"]}')
            else:
                print('Cant afford the minimum quantity')

        elif self.cache[asset]['type'] == 'SHORT' and self.short_confirmed(asset):

            aux = self.affordable(asset)
            if aux['affordable']:

                if self.demo:
                    order = {'orderId': self.demoId, 'status': 'NEW', 'price': aux['price'], 'executedQty': aux['qty'], 'updateTime': int(time.time() * 1000)}
                    self.demoId += 1
                else:
                    order = client.futures_create_order(symbol=asset, side='SELL', type='LIMIT', price=aux['price'], quantity=aux['qty'], timeInForce='GTC')

                self.cache[asset]['type'] = 'SHORT'
                self.cache[asset]['direction'] = 'IN'
                self.tp_price[asset] = self.mid_price * (1 - self.tp)
                self.sl_price[asset] = self.mid_price * (1 + self.sl)
                self.cache[asset]['order'] = order
                print('Sell order sent')
            else:
                print('Cant afford the minimum quantity')

    def close_position(self, asset):

        if self.cache[asset]['type'] == 'LONG':

            if self.demo:
                self.cache[asset]['order']['status'] = 'NEW'
                self.cache[asset]['direction'] = 'OUT'

            else:
                order = client.futures_create_order(symbol=asset, side='SELL', type='LIMIT', price=self.prices[asset]['ask'], quantity=self.cache[asset]['order']['executedQty'], timeInForce='GTC')
                self.cache[asset]['order'] = order
                self.cache[asset]['direction'] = 'OUT'
                print('Closing Long Position')

        if self.cache[asset]['type'] == 'SHORT':

            if self.demo:
                self.cache[asset]['order']['status'] = 'NEW'
                self.cache[asset]['direction'] = 'OUT'
            else:
                order = client.futures_create_order(symbol=asset, side='BUY', type='MARKET', price=self.prices[asset]['bid'], quantity=self.cache[asset]['order']['executedQty'], timeInForce='GTC')
                self.cache[asset]['order'] = order
                self.cache[asset]['direction'] = 'OUT'
            print('Closing Short Position')

    def abort(self, asset):

        if self.cache[asset]['direction'] == 'IN':

            if self.demo:
                self.cache[asset]['order']['status'] = 'CANCELED'
            else:
                self.cache[asset]['order'] = client.futures_cancel_order(symbol=asset, orderId=self.cache[asset]['order']['orderId'])

                print('Cancelling Order')

                if self.cache[asset]['order']['status'] != 'CANCELED':
                    print('Error cancelling order')
                    raise KeyboardInterrupt
                else:
                    self.cache[asset] = {'type': None, 'direction': None, 'order':{'status': 'NONE'}}
                    print('Order Canceled because of taking too much time')

            return None

        if self.cache[asset]['direction'] == 'OUT':

            if self.demo:
                self.cache[asset]['order']['status'] = 'FILLED'
            else:
                client.futures_cancel_order(symbol=asset, orderId=self.cache[asset]['order']['orderId'])
                if self.cache[asset]['type'] == 'LONG':
                    self.cache[asset]['order'] = client.futures_create_order(symbol=asset, side='SELL', type='MARKET', quantity=self.cache[asset]['order']['executedQty'], timeInForce='GTC')
                elif self.cache[asset]['type'] == 'SHORT':
                    self.cache[asset]['order'] = client.futures_create_order(symbol=asset, side='BUY', type='MARKET', quantity=self.cache[asset]['order']['executedQty'], timeInForce='GTC')

                print('Creating Market Order')
                self.out_counter[asset] = 0

                if self.cache[asset]['order']['status'] != 'FILLED':
                    print('Error filling/cancelling order')
                    raise KeyboardInterrupt
                else:
                    self.cache[asset] = {'type': None, 'direction': None, 'order': {'status': 'NONE'}}
                    print('Order forced to market because of taking too much time')

            return None

    def new_data(self, depth_cache):

        asset = depth_cache.symbol
        self.data[asset].append({'t': depth_cache.update_time, 'a': depth_cache.get_asks()[:self.limit], 'b': depth_cache.get_bids()[:self.limit]})

        if self.filling_download(asset):
            self.counter[asset] += 1
        else:
            self.save_download(asset)

        if self.filling_buffer(asset):
            self.fill_buffer(asset, depth_cache)

        else:

            if self.missing_scalers(asset):
                self.fill_scalers(asset)

            else:

                self.update_buffer(asset, depth_cache)

                if self.confirmed_position(asset):

                    # print(f'1 Position Open. PNL: {(float(self.mid_price) / float(self.cache[asset]["order"]["price"]) - 1) * 100}')
                    if self.position_expiry(asset):
                        self.close_position(asset)
                    else:
                        if self.hit_sl(asset):
                            print(f'SL Hitted at {self.cache[asset]["order"]["price"]}')
                            self.close_position(asset)
                        else:
                            if self.hit_tp(asset):
                                print(f'TP Hitted at {self.cache[asset]["order"]["price"]}')
                                if self.trailing > 0:
                                    self.update_trailing(asset)
                                else:
                                    self.close_position(asset)

                else:

                    if self.no_position(asset):
                        if self.signal(asset):
                            self.open_position(asset)

                    elif self.pending_position(asset):
                        print('Check 1')
                        if self.order_expiry(asset):
                            print('Check 2')
                            self.abort(asset)


if __name__ == "__main__":
    trader = Trader(assets=['BTCUSDT'], model_date='2021_03_04', treshold=0.75, cache='../data/cache.pickle', stats='../data/stats.pickle',
                 reset_stats=True, reset_cache=True, demo=0, limit_depth=10, window=100, limit_file=3600, leverage=5,
                 size=1, confirm=1, max_in_count=10, max_out_count=10, expiry_position=300, expiry_order=10, trailing=0.0001, sl=0.01, tp=0.001)

    dcm = DepthCacheManager(client, 'BTCUSDT', callback=trader.new_data, refresh_interval=3600, limit=50, bm=bm)
