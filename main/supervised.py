import pandas as pd
import pickle
import numpy as np


def create_supervised_data(symbols, dates, window=100, pnl_window=100, pnl=0.5, step=1):

    for symbol in symbols:

        for d in dates:

            path = f'../data/{symbol.upper()}/{d}'

            df = pd.read_pickle(f'{path}/lob_norm.pickle')
            df_raw = pd.read_pickle(f'{path}/lob_raw.pickle')

            target = pd.DataFrame()
            target['p'] = (df_raw['pa0'] + df_raw['pb0']) / 2
            target['mb'] = target['p'].rolling(pnl_window).mean()
            target['ma'] = target['p'].rolling(pnl_window).mean().shift(-pnl_window)
            target['buy'] = (target['ma'] > target['mb'] * (1 + pnl)).astype(int)
            target['sell'] = (target['ma'] < target['mb'] * (1 - pnl)).astype(int)
            target['wait'] = (target['buy'] + target['sell'] == 0).astype(int)
            target = target.dropna()

            print(target[['buy', 'sell', 'wait']].dropna().value_counts())

            x, y = list(), list()
            df = df.values
            target = target[['buy', 'sell', 'wait']].values

            for i in range(pnl_window + window, len(df) - pnl_window, step):
                x.append(df[i-window:i])
                y.append(target[i])

            with open(f'{path}/supervised.pickle', 'wb') as file:
                pickle.dump((np.array(x).reshape((len(x), window, df.shape[1], 1)), np.array(y)), file)

            print(f'Supervised Data Created: {path}')


def merge(symbols, dates, name):
    for symbol in symbols:
        data = [pickle.load(open(f'../data/{symbol.upper()}/{d}/supervised.pickle', 'rb')) for d in dates]
        x = np.concatenate([data[i][0] for i in range(len(dates))])
        y = np.concatenate([data[i][1] for i in range(len(dates))])
        with open(f'../data/{symbol.upper()}/{name}/supervised.pickle', 'wb') as file:
            pickle.dump((x, y), file)


if __name__ == "__main__":
    create_supervised_data(['BTCUSDT'], ['2021_03_12', '2021_03_13'], window=512, pnl_window=240, pnl=0.001, step=1)