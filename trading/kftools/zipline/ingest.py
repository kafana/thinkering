import pandas as pd
from iexfinance import Stock


def _load_from_iex(symbol, start, end, range='5y'):
    data = Stock(symbol).get_chart(range=range)
    if not data:
        raise RuntimeError('Unable to get data from IEX '
                           'for symbol {} and {} range'.format(symbol, range))
    df = pd.DataFrame.from_dict(data)
    df = df[['date', 'open', 'close', 'high', 'low', 'volume']]
    df['price'] = df['open'] # Keep this because most quantopian algorithms use `price` column
    df['date'] = pd.to_datetime(df['date'])
    df = df.reset_index(drop=True)
    df = df.set_index('date')
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


providers = {
    'iex': _load_from_iex
}


def load_from(symbols, start=None, end=None, provider='iex', range=5):
    if provider not in providers:
        raise ValueError("Invalid data provider '{}'".format(provider))
    data = {}
    for symbol in symbols:
        data[symbol] = providers[provider](symbol, start, end, range='{}y'.format(range))
    panel = pd.Panel(data)
    return panel
