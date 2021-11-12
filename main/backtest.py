import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def load_prices(asset, test_date, window=100, pnl_window=100):
    df = pd.read_pickle(f'../data/{asset}/{test_date}/lob_raw.pickle')
    prices = (df['pa0'] + df['pb0']) / 2
    return prices.iloc[window + pnl_window: -pnl_window]


class Backtest:

    def __init__(self, assets, model_date, test_date, fee=0.00036, leverage=10, window=100, pnl_window=100, verbose=2):

        self.model = {asset: load_model(f'../data/{asset}/{model_date}/model.h5') for asset in assets}
        self.prices = {asset: load_prices(asset, test_date, window, pnl_window) for asset in assets}
        self.data = {asset: pickle.load(open(f'../data/{asset}/{test_date}/supervised.pickle', 'rb'))[0] for asset in assets}
        self.predictions = {asset: self.model[asset].predict(self.data[asset]) for asset in assets}
        self.assets = assets
        self.leverage = leverage
        self.fee = fee
        self.verbose = verbose

        self.positions = {asset: {'type': 0, 'price': 0} for asset in assets}
        self.sl = {asset: 0 for asset in assets}
        self.tp = {asset: 0 for asset in assets}
        self.portfolio = 0
        self.pnl = 0
        self.confirm_short = 0
        self.confirm_long = 0
        self.trades = 0
        self.expiry = {asset: 0 for asset in assets}
        self.right_tp = 0
        self.wrong_sl = 0
        self.right_expiry = 0
        self.right_trailing = 0
        self.wrong_expiry = 0
        self.wrong_trailing = 0

    def initialize_parameters(self):
        self.positions = {asset: {'type': 0, 'price': 0} for asset in self.assets}
        self.sl = {asset: 0 for asset in self.assets}
        self.tp = {asset: 0 for asset in self.assets}
        self.portfolio = 0
        self.pnl = 0
        self.confirm_short = 0
        self.confirm_long = 0
        self.trades = 0
        self.expiry = {asset: 0 for asset in self.assets}
        self.right_tp = 0
        self.wrong_sl = 0
        self.right_expiry = 0
        self.right_trailing = 0
        self.wrong_expiry = 0
        self.wrong_trailing = 0

    def long(self, asset, t, sl=0.001, tp=0.001):
        price = self.prices[asset].iloc[t]
        self.sl[asset] = price * (1 - sl)
        self.tp[asset] = price * (1 + tp)
        self.positions[asset] = {'type': 1, 'price': price}
        if self.verbose == 2:
            print(f'{t} - Opening Long at {price}')

    def short(self, asset, t, sl=0.001, tp=0.001):
        price = self.prices[asset].iloc[t]
        self.sl[asset] = price * (1 + sl)
        self.tp[asset] = price * (1 - tp)
        self.positions[asset] = {'type': -1, 'price': price}
        if self.verbose == 2:
            print(f'{t} - Opening Short at {price}')

    def close(self, asset, t, trailing, expiry=100):
        price = self.prices[asset].iloc[t]
        self.expiry[asset] += 1

        if self.expiry[asset] >= expiry:
            if self.positions[asset]['type'] == 1:
                pnl = (price / self.positions[asset]['price'] - 1 - self.fee) * self.leverage * (1 / len(self.assets))
            else:
                pnl = (1 - price / self.positions[asset]['price'] - self.fee) * self.leverage * (1 / len(self.assets))
            self.portfolio = self.portfolio * (1 + pnl)
            if self.verbose == 2:
                print(f'{t} - Expiry at {price}. PNL: {pnl * 100:.2f}%. Portfolio: {self.portfolio:.2f}$')
            self.positions[asset] = {'type': 0, 'price': 0}
            self.expiry[asset] = 0
            if pnl > 0:
                self.right_expiry += 1
            else:
                self.wrong_expiry += 1
            return True

        if self.positions[asset]['type'] == 1:

            # Long TP
            if price >= self.tp[asset]:
                if trailing > 0:
                    self.expiry[asset] = 0
                    self.tp[asset] = price * (1 + trailing)
                    self.sl[asset] = price * (1 - trailing)
                    return False

                else:
                    self.right_tp += 1
                    pnl = (price / self.positions[asset]['price'] - 1 - self.fee) * self.leverage * (1 / len(self.assets))
                    self.portfolio = self.portfolio * (1 + pnl)
                    if self.verbose == 2:
                        print(f'{t} - Closing Long at {price} with TP. PNL: {pnl * 100:.2f}%. Portfolio: {self.portfolio:.2f}$')
                    self.positions[asset] = {'type': 0, 'price': 0}
                    self.expiry[asset] = 0
                    return True

            # Long SL
            elif price <= self.sl[asset]:
                pnl = (price / self.positions[asset]['price'] - 1 - self.fee) * self.leverage * (1 / len(self.assets))

                if pnl < 0 and trailing > 0:
                    self.wrong_trailing += 1
                elif pnl < 0 and trailing == 0:
                    self.wrong_sl += 1
                elif pnl > 0 and trailing > 0:
                    self.right_trailing += 1

                self.portfolio = self.portfolio * (1 + pnl)
                if self.verbose == 2:
                    print(f'{t} - Closing Long at {price} with SL. PNL: {pnl * 100:.2f}%. Portfolio: {self.portfolio:.2f}$')
                self.positions[asset] = {'type': 0, 'price': 0}
                self.expiry[asset] = 0
                return True

        if self.positions[asset]['type'] == -1:

            # Short TP
            if price <= self.tp[asset]:
                if trailing > 0:
                    self.expiry[asset] = 0
                    self.tp[asset] = price * (1 + trailing)
                    self.sl[asset] = price * (1 - trailing)
                    return False

                else:
                    self.right_tp += 1
                    pnl = (1 - price / self.positions[asset]['price'] - self.fee) * self.leverage * (1 / len(self.assets))
                    self.portfolio = self.portfolio * (1 + pnl)
                    if self.verbose == 2:
                        print(f'{t} - Closing Short at {price} with TP. PNL: {pnl * 100:.2f}%. Portfolio: {self.portfolio:.2f}$')
                    self.positions[asset] = {'type': 0, 'price': 0}
                    self.expiry[asset] = 0
                    return True

            # Short SL
            elif price >= self.sl[asset]:
                pnl = (1 - price / self.positions[asset]['price'] - self.fee) * self.leverage * (1 / len(self.assets))

                if pnl < 0 and trailing > 0:
                    self.wrong_trailing += 1
                elif pnl < 0 and trailing == 0:
                    self.wrong_sl += 1
                elif pnl > 0 and trailing > 0:
                    self.right_trailing += 1

                self.portfolio = self.portfolio * (1 + pnl)
                if self.verbose == 2:
                    print(f'{t} - Closing Short at {price} with SL. PNL: {pnl * 100:.2f}%. Portfolio: {self.portfolio:.2f}$')
                self.positions[asset] = {'type': 0, 'price': 0}
                self.expiry[asset] = 0
                return True
        return False

    def signal(self, asset, t, confirm=5, treshold=0.9):

        prediction = self.predictions[asset][t]

        if np.max(prediction) > treshold:
            prediction = np.argmax(prediction)
        else:
            prediction = 2

        if prediction == 0:
            if self.confirm_long >= confirm:
                self.confirm_long = 0
                return 1
            self.confirm_long += 1
            self.confirm_short = 0
            return None

        elif prediction == 1:
            if self.confirm_short >= confirm:
                self.confirm_short = 0
                return -1
            self.confirm_short += 1
            self.confirm_long = 0
            return None

        else:
            self.confirm_short = 0
            self.confirm_long = 0
            return 0

    def backtest(self, step=1, confirm=5, sl=0.002, tp=0.002, trailing=0, treshold=0.9, expiry=100, portfolio=100):

        self.initialize_parameters()
        self.portfolio = portfolio

        long_time, long_price = {asset: list() for asset in self.assets}, {asset: list() for asset in self.assets}
        short_time, short_price = {asset: list() for asset in self.assets}, {asset: list() for asset in self.assets}
        close_time, close_price = {asset: list() for asset in self.assets}, {asset: list() for asset in self.assets}
        times = {asset: [t for t in range(len(self.prices[asset]))] for asset in self.assets}
        prices = self.prices

        for asset in self.assets:

            for t in range(0, len(self.prices[asset]), step):

                if self.positions[asset]['type'] == 0:
                    signal = self.signal(asset, t, confirm, treshold)
                    if signal == 1:
                        self.long(asset, t, sl, tp)
                        long_time[asset].append(t)
                        long_price[asset].append(self.prices[asset].iloc[t])

                    elif signal == -1:
                        self.short(asset, t, sl, tp)
                        short_time[asset].append(t)
                        short_price[asset].append(self.prices[asset].iloc[t])

                elif self.close(asset, t, trailing, expiry):
                    close_time[asset].append(t)
                    close_price[asset].append(self.prices[asset].iloc[t])
                    self.trades += 1

        if self.verbose == 2:
            fig, ax = plt.subplots()
            for asset in self.assets:
                ax.plot(times[asset], prices[asset], color='gray')
                ax.plot(long_time[asset], long_price[asset], ls='', marker='o', color='green', markersize=5)
                ax.plot(short_time[asset], short_price[asset], ls='', marker='o', color='red', markersize=5)
                ax.plot(close_time[asset], close_price[asset], ls='', marker='x', color='blue', markersize=5)
                plt.show()
        if self.verbose >= 1:
            print(f'Wrong SL: {self.wrong_sl}. Right TP: {self.right_tp}. Wrong Expiry: {self.wrong_expiry}. '
                  f'Right Expiry: {self.right_expiry}. Wrong Trailing: {self.wrong_trailing}. '
                  f'Right Trailing: {self.right_trailing}. Trades:{self.trades}. Portfolio: {self.portfolio}.')

        return self.portfolio, self.trades


bt = Backtest(['BTCUSDT'], model_date='Aggregated', test_date='2021_03_16', fee=0.0003, leverage=5, window=100, pnl_window=100, verbose=2)
print(bt.backtest(step=1, confirm=0, sl=0.05, tp=0.0025, treshold=0.9, trailing=0.00025, expiry=500))