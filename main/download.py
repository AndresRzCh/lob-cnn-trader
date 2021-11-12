from binance.websockets import BinanceSocketManager
from binance.depthcache import DepthCacheManager
from binance.client import Client
from datetime import date
import pickle
import os

# Binance API Key and Settings:
api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')
client = Client(api_key, secret_key)
bm = BinanceSocketManager(client)


class Downloader:

    def __init__(self, symbol, limit_file=3600, limit=100, initial=None):  # Limit file to 3600 rows with a depth of 100
        self.path = f'../data/{symbol.upper()}/{date.today().strftime("%Y_%m_%d")}'
        self.limit_file = limit_file
        self.data = list()
        self.counter = 0
        self.limit = limit
        self.symbol = symbol

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not initial:
            files = list(filter(lambda k: 'LOB' in k, os.listdir(self.path)))
            self.n = len(files)
        else:
            self.n = initial

    def update(self, dc):
        self.data.append({'t': dc.update_time, 'a': dc.get_asks()[:self.limit], 'b': dc.get_bids()[:self.limit]})

        if self.counter < self.limit_file:
            self.counter += 1
        else:
            self.counter = 0
            path = f'../data/{self.symbol.upper()}/{date.today().strftime("%Y_%m_%d")}'
            if path != self.path:
                os.makedirs(path)
                self.path = path
                self.n = 0
            filename = f'{self.path}/LOB{str(self.n).zfill(3)}.pickle'
            with open(filename, 'wb') as file:
                pickle.dump(self.data, file)
            print(f'Saved Data at: {filename}')
            self.n += 1
            self.data = list()


if __name__ == "__main__":
    dcm = {}
    downloaders = {}
    for s in ['BTCUSDT']:
        downloaders[s] = Downloader(s, limit_file=3600, limit=50)
        dcm[s] = DepthCacheManager(client, s, callback=downloaders[s].update, refresh_interval=3600, limit=50, bm=bm)