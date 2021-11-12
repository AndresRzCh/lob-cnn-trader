from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import os


def reduce(lob, limit):
    return [np.insert(np.array(f['a'][:limit][::-1] + f['b'][:limit]).flatten('F'), 0, f['t']) for f in lob]


def clean_raw_data(symbols, dates, limit=10, pnl_window=100, included_files=None, scale=None):
    for symbol in symbols:

        for d in dates:

            path = f'../data/{symbol.upper()}/{d}'

            if included_files is None:
                files = list(filter(lambda k: 'LOB' in k, os.listdir(path)))
            else:
                files = included_files

            aux = list()

            for file in files:
                aux.append(pd.DataFrame(reduce(lob=pickle.load(open(path + f'/{file}', 'rb')), limit=limit)))

            df = pd.concat(aux)
            df = df.set_index(0)
            a = np.arange(limit, 0, -1) - 1
            b = np.arange(0, limit, 1)
            df.columns = [f'pa{i}' for i in a] + [f'pb{i}' for i in b] + [f'qa{i}' for i in a] + [f'qb{i}' for i in b]

            df.to_pickle(path + '/lob_raw.pickle')

            # Standard Scaler
            if scale is None:
                df_unscaled = df.iloc[:pnl_window, :]
            else:
                df_unscaled = pd.read_pickle(f'../data/{symbol}/{scale}/lob_raw.pickle')

            scaler = StandardScaler()
            scaler.fit(df_unscaled)
            df = pd.DataFrame(scaler.transform(df.values), columns=df.columns)
            df.to_pickle(path + '/lob_norm.pickle')

            print(f'Folder converted: {path}')


if __name__ == "__main__":
    clean_raw_data(['BTCUSDT'], ['2021_03_12', '2021_03_13'], included_files=None, limit=10, pnl_window=240, scale='2021_03_12')
