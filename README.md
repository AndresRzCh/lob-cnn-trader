# LOB CNN Trader
This is a implementation of the theory from 
the paper of Tsantekidis _et al_ [(Forecasting Stock Prices from the Limit Order
Book using Convolutional Neural Networks)](https://ieeexplore.ieee.org/document/8010701) 
to Cryptocurrency Markets using Binance Trading API.

### Dependencies
* [Binance API Python Wrapper](https://python-binance.readthedocs.io/en/latest/)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)

### Scripts
There are seven scripts at `/main` with different utilities:
* `download.py` is used to real-time download the Limit Order Book using websockets.
* `refactor.py` prepares the data to be used in supervised learning training 
* `supervised.py` split the data randomly to prepare the supervised learning datasets
* `train.py` implements the Deep Learning model from the paper and trains it with the provided data
* `backtest.py` perform tests based on historical data to check the performance of the model
* `main.py` uses a pre-trained model to live trade using Binance API
* `optimize.py` is a set of helper functions to find the best model and parameter tuning

### Disclaimer
There are no warranties expressed or implied in this repository. I am not responsible for anything done with this program. You assume all responsibility and liability. Use it at your own risk.  
