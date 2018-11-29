
import sys
from datetime import timedelta
from datetime import datetime

from dateutil.relativedelta import relativedelta
from iexfinance import Stock
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import deque
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas_datareader as pdr

from kftools.optimizations import PortfolioOptimizer


def read_stock_data(symbols, start_date, end_date, provider='yahoo'):
    stocks = {}
    if not isinstance(symbols, (list, tuple,)):
        symbols = [symbols]
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    for symbol in symbols:
        stocks[symbol] = pdr.DataReader(symbol, provider,
                                        start_date, end_date)
        stocks[symbol].drop(['Close'], axis=1, inplace=True)
        stocks[symbol].rename(str.lower, axis='columns', inplace=True)
        stocks[symbol].rename(columns={'adj close': 'close'}, inplace=True)
        stocks[symbol].index.rename('date', inplace=True)
    return stocks

def rolling_window(array, length):
    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] <= length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = (orig_shape[0] - length + 1)
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    return as_strided(array, new_shape, new_strides)


class Factor(object):

    def __init__(self, data, window_length=15, inputs='close', preserve_data=False):
        """
        window_length :
        Length of the lookback window over which to compute factor.
        """
        if preserve_data:
            self.data = data
        else:
            self.data = data[inputs].sort_index(ascending=True).tail(window_length)

    @classmethod
    def exponential_weights(cls, length, decay_rate):
        x = np.full(length, decay_rate, np.dtype('float64')) ** np.arange(length + 1, 1, -1)
        return x


class Returns(Factor):
    """
    Calculates the percent change in close price over the given window_length
    """
    def compute(self):
        out = (self.data[-1] - self.data[0]) / self.data[0]
        return out


class RSI(Factor):
    """
    Relative Strength Index
    """
    def compute(self):
        diffs = np.diff(self.data, axis=0)
        ups = np.nanmean(np.clip(diffs, 0, np.inf), axis=0)
        downs = np.abs(np.nanmean(np.clip(diffs, -np.inf, 0), axis=0))
        return 100 - (100 / (1 + (ups / downs)))


class MA(Factor):
    """
    Moving Average
    """
    def compute(self, window=10):
        return np.nanmean(self.data.rolling(window=window).mean())


class EWMA(Factor):
    """
    Exponential Weighted Moving Average
    """
    def compute(self, span=30):
        return np.nanmean(self.data.ewm(span=span).mean())


class BollingerBands(Factor):
    """
    Bollinger Bands technical indicator.
    k : float
        The number of standard deviations to add or subtract to create the
        upper and lower bands.
    """

    def compute(self, k=2):
        out = {}
        difference = k * np.nanstd(self.data, axis=0)
        out['middle'] = middle = np.nanmean(self.data, axis=0)
        out['upper']= middle + difference
        out['lower'] = middle - difference
        return out


class MACD(Factor):
    """
    Moving Average Convergence/Divergence (MACD) Signal line
    """
    def __init__(self, data, fast_period=12, slow_period=26, signal_period=9, inputs='close'):
        if slow_period <= fast_period:
            raise ValueError("'slow_period' must be greater than 'fast_period'")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        window_length = slow_period + signal_period - 1
        self.data = data[inputs].sort_index(ascending=True).tail(window_length)

    def _ewma(self, data, length, axis=1):
        decay_rate = 1.0 - (2.0 / (1.0 + length))
        return np.average(
            data,
            axis=axis,
            weights=Factor.exponential_weights(length, decay_rate)
        )

    def compute(self):
        slow_EWMA = self._ewma(
            rolling_window(self.data, self.slow_period),
            self.slow_period
        )
        fast_EWMA = self._ewma(
            rolling_window(self.data, self.fast_period)[-self.signal_period:],
            self.fast_period
        )
        macd = fast_EWMA - slow_EWMA
        return self._ewma(macd.T, self.signal_period, axis=0)


class Context(object):

    def __init__(self, portfolio=[], benchmark_symbol='SPY'):
        self.portfolio = portfolio if isinstance(portfolio, (list, tuple)) else [portfolio]
        self.technical_indicator_states = {}
        self.window_length = 7 # Number of data points to collect before updating train collections
        self.benchmark = deque(maxlen=self.window_length)
        self.benchmark_symbol = benchmark_symbol
        self.features = ['RSI','EMA','MACD','SMA_5','SMA_10','bb_lower','bb_middle','bb_upper']
        self.response = ['Class']
        self.X = pd.DataFrame(columns=self.features) # X train data
        self.Y = pd.DataFrame(columns=self.response) # Y train data
        self.prediction = {} # Stores most recent prediction
        self.day_counter = 0
        self.position_adjustment_days = 5 # Number of days to wait before adjusting positions
        self.min_data_points = 500
        self.max_data_points = 1500
        self.total_buy = 0
        self.positions = []
        self.data = {}
        self.benchmark_data = {}
        self.output = None
        self.today = pd.to_datetime((pd.datetime.today() - BDay(1)).strftime('%Y-%m-%d'))
        self.classifier = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10)

    def _add_data_for_symbol(self, symbol, container, range, provider):
        if provider == 'iex':
            data = Stock(symbol).get_chart(range='{}y'.format(range))
            if not data:
                raise RuntimeError('Unable to get data from IEX for symbol {} and {} range'.format(symbol, range))
            container[symbol] = pd.DataFrame.from_dict(data)
            container[symbol]['date'] = pd.to_datetime(container[symbol]['date'])
            container[symbol].reset_index(drop=True, inplace=True)
            container[symbol] = container[symbol].set_index('date')
        else:
            start_date = (datetime.today() - relativedelta(years=range)).strftime('%Y-%m-%d')
            end_date = datetime.today().strftime('%Y-%m-%d')
            stocks = read_stock_data(symbol, start_date, end_date, provider=provider)
            container[symbol] = stocks[symbol]

    def _load_portfolio_data(self, provider='iex', range=5):
        for symbol in self.portfolio:
            self._add_data_for_symbol(symbol, self.data, range, provider)
        self._add_data_for_symbol(self.benchmark_symbol, self.benchmark_data, range, provider)

    @property
    def _adjust_position(self):
        return True if self.day_counter % self.position_adjustment_days == 0 else False

    def _increment_current_day(self, end_date):
        end = end_date.strftime('%Y-%m-%d')

        try:
            iloc = self.benchmark_data[self.benchmark_symbol].index.get_loc(end)
        except KeyError:
            return False

        stop = self.benchmark_data[self.benchmark_symbol].index[iloc]
        prev = stop if iloc == 0 else self.benchmark_data[self.benchmark_symbol].index[iloc - 1]
        next = stop if iloc == self.benchmark_data[self.benchmark_symbol].index.size - 1 else \
            self.benchmark_data[self.benchmark_symbol].index[iloc + 1]

        self.today = stop
        self.day_counter += 1

        return True

    def _add_to_portfolio(self, prediction_array):
        # Note: We should be able to run prediction_array > 0 condition
        return True if prediction_array.size == 1 and prediction_array[0] else False

    def _get_current_price_for_symbol(self, symbol, column='open'):
        # TODO: This fn needs to get 1h data
        if symbol in self.data:
            return self.data[symbol].loc[self.today][column]
        if symbol in self.benchmark_data:
            return self.benchmark_data[symbol].loc[self.today][column]
        return None

    def _get_history(self, symbols=None, column='close', days=30, frequency='1d'):
        data = pd.DataFrame()
        end = self.today.strftime('%Y-%m-%d')
        symbols = symbols if isinstance(symbols, (list, tuple)) else [symbols]
        for symbol in symbols:
            if symbol in self.data:
                iloc = self.data[symbol].index.get_loc(end)
                start = self.data[symbol].index[iloc - days].strftime('%Y-%m-%d')
                data[symbol] = self.data[symbol][start:end][column]
        return data

    def _load_state(self, start_date, end_date):
        data = []
        index = ['Name']
        returns = ['returns_5']

        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')

        for key, value in self.data.items():
            features = {key: 0.0 if key not in index else '' for key in index + self.features + returns}
            features['Name'] = key
            features['SMA_5'] = MA(value.loc[start:end], window_length=5).compute(window=5)
            features['SMA_10'] = MA(value.loc[start:end], window_length=10).compute(window=10)
            features['EMA'] = EWMA(value.loc[start:end], window_length=30).compute(span=30)
            features['RSI'] = RSI(value.loc[start:end], window_length=15).compute()
            features['MACD'] = MACD(value.loc[start:end]).compute()
            bb = BollingerBands(value.loc[start:end], window_length=20).compute(k=2)
            features['bb_lower'] = bb['lower']
            features['bb_middle'] = bb['middle']
            features['bb_upper'] = bb['upper']
            features['returns_5'] = Returns(value.loc[start:end], window_length=5).compute()

            data.append(features)

        self.output = pd.DataFrame(data, columns=index + self.features + returns)
        self.output.reset_index(drop=True, inplace=True)
        self.output = self.output.set_index(index[0])
        self.output = self.output.dropna()

    def _is_benchamrk_ready(self):
        return True if len(self.benchmark) == self.benchmark.maxlen else False

    def _is_symbol_ready(self, symbol):
        return True if len(self.technical_indicator_states[symbol].index) == self.window_length else False

    def _rebalance(self):
        for symbol, _ in self.output.iterrows():
            if symbol in self.technical_indicator_states:
                self.technical_indicator_states[symbol] = self.technical_indicator_states[symbol].append(
                    self.output.loc[symbol], ignore_index=True)
            else:
                self.technical_indicator_states[symbol] = pd.DataFrame(
                    self.output.loc[symbol]).transpose()

        self.benchmark.append(self._get_current_price_for_symbol(self.benchmark_symbol, 'open'))

        # Wait till we accumulate enough data inside of the benchmark collection
        if self._is_benchamrk_ready():

            # Calculate Benchmark return
            # benchmark = (self.benchmark[-1] - self.benchmark[0]) / self.benchmark[0]
            benchmark = Returns(self.benchmark, preserve_data=True).compute()
            symbol_X_tests = {}

            for symbol, _ in self.output.iterrows():
                # Make sure there is enough data in for the symbol
                if self._is_symbol_ready(symbol):
                    # Take the last return and check if it beat benchmark
                    returns_5 = self.technical_indicator_states[symbol].iloc[-1]['returns_5']
                    change = returns_5 > benchmark and returns_5 > 0

                    Y_train = {}
                    Y_train['Class'] = change

                    # Load X train data from the 1st row
                    X_train = {}
                    for column in self.technical_indicator_states[symbol].columns:
                        if column in self.features:
                            X_train[column] = self.technical_indicator_states[symbol].iloc[0][column]

                    self.X = self.X.append([X_train], ignore_index=True)
                    self.Y = self.Y.append([Y_train], ignore_index=True)

                    # Load X test data from the latest row
                    symbol_test = {}
                    for column in self.technical_indicator_states[symbol].columns:
                        if column in self.features:
                            symbol_test[column] = self.technical_indicator_states[symbol].iloc[-1][column]

                    symbol_X_tests[symbol] = symbol_test

                    # Purge 1st row
                    self.technical_indicator_states[symbol] = self.technical_indicator_states[symbol].iloc[1:]

            # There needs to be enough data points to make a good model and adjust positions once per 5 days
            if len(self.X.index) >= self.min_data_points and self._adjust_position:

                # Purge data if we reach max number of endpoints (do we have to to this?)
                if len(self.X.index) > self.max_data_points:
                    self.X = self.X.iloc[self.min_data_points:]
                    self.Y = self.Y.iloc[self.min_data_points:]

                self.Y[self.response] = self.Y[self.response].astype('bool')
                self.classifier.fit(self.X[self.features].values, self.Y[self.response].values) # Generate the model

                self.total_buy = 0
                for symbol, _ in self.output.iterrows():
                    if symbol in symbol_X_tests:
                        X_test = pd.DataFrame(columns=self.features)
                        X_test = X_test.append(symbol_X_tests[symbol], ignore_index=True)
                        self.prediction[symbol] = self.classifier.predict(X_test)
                        self.total_buy += 1 if self.prediction[symbol][0] else 0

                position_symbols = []
                for symbol, _ in self.output.iterrows():
                    if symbol in self.prediction:
                        if self._add_to_portfolio(self.prediction[symbol]):
                            position_symbols.append(symbol)
                        else:
                            print("SELL - order_target_percent(", symbol, ", 0)")

                if self.total_buy != 0:
                    self.positions = self._get_history(position_symbols, 'close', 30, '1d')
                    weights1 = PortfolioOptimizer.optimize_weights1(self.positions)
                    self.positions = self._get_history(position_symbols, 'close', 100, '1d')
                    weights2, _, _ = PortfolioOptimizer.optimize_weights2(self.positions)
                    for symbol in weights1.keys():
                        print("BUY - order_target_percent(", symbol, weights[symbol], ")")


def main():
    n = 150
    # from_date = pd.to_datetime('2015-10-01') + BDay(1)
    from_date = pd.to_datetime('2017-11-01') + BDay(1)
    # to_date = pd.to_datetime('2015-12-30') + BDay(1)
    to_date = pd.to_datetime('2018-01-30') + BDay(1)
    # context = Context(portfolio=['AAPL', 'ADSK', 'ADBE', 'ADI', 'AMAT', 'AMD', 'APH', 'ARW', 'AVT', 'MSFT', 'INTC', 'AMZN', 'GOOG'])
    context = Context(portfolio=['IBM', 'SBUX', 'XOM', 'AAPL', 'MSFT', 'TLT', 'SHY', 'CVS', 'AMZN', 'GOOG', 'AMD'])
    context._load_portfolio_data(provider='yahoo')
    for i in range(n):
        if context._increment_current_day(to_date):
            context._load_state(from_date, to_date)
            context._rebalance()
        to_date = pd.to_datetime(to_date.to_pydatetime() + BDay(1))
        from_date = pd.to_datetime(from_date.to_pydatetime() + BDay(1))
        print("Processing dates from", from_date, "to", to_date)

if __name__ == "__main__":
    main()
