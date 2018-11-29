import sys
from datetime import datetime

import zipline
from zipline.api import (
    set_slippage, 
    slippage,
    set_commission, 
    commission, 
    order_target_percent,
    symbols,
    symbol,
    record
)
from zipline import TradingAlgorithm
from zipline.assets import Equity
from zipline.utils.run_algo import run_algorithm
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pylab as plt
import pyfolio as pf

from kftools.optimizations import PortfolioOptimizer
from kftools.zipline.ingest import load_from

ASSETS = ['IBM', 'SBUX', 'XOM', 'AAPL', 'MSFT']

def initialize(context):
    '''
    Called once at the very beginning of a backtest (and live trading). 
    Use this method to set up any bookkeeping variables.
    
    The context object is passed to all the other methods in your algorithm.

    Parameters

    context: An initialized and empty Python dictionary that has been 
             augmented so that properties can be accessed using dot 
             notation as well as the traditional bracket notation.
    
    Returns None
    '''
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))
    context.tick = 0
    context.assets = symbols(*ASSETS)
    
def handle_data(context, data):
    '''
    Called when a market event occurs for any of the algorithm's 
    securities. 

    Parameters

    data: A dictionary keyed by security id containing the current 
          state of the securities in the algo's universe.

    context: The same context object from the initialize function.
             Stores the up to date portfolio as well as any state 
             variables defined.

    Returns None
    '''
    
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < 100:
        return
    # Get rolling window of past prices and compute returns
    prices = data.history(context.assets, 'price', 100, '1d').dropna()
    try:
        # Perform Markowitz-style portfolio optimization
        weights, _, _ = PortfolioOptimizer.optimize_weights2(prices)
        # Rebalance portfolio accordingly
        for stock, weight in weights.items():
            order_target_percent(stock, weight)
    except ValueError as e:
        # Sometimes this error is thrown
        # ValueError: Rank(A) < p or Rank([P; A; G]) < n
        pass

    # Record some values for later inspection
    for asset in context.assets:
        short_ma = data.history(asset, 'price', bar_count=50, frequency="1d").mean()
        long_ma = data.history(asset, 'price', bar_count=100, frequency="1d").mean()
        kwargs = {
            asset.symbol: data.current(stock, 'price'),
            '{}_SMA'.format(asset.symbol): short_ma,
            '{}_LMA'.format(asset.symbol): long_ma
        }
        record(**kwargs)


def analyze(context, results):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')

    ax2 = fig.add_subplot(212)
    results['AAPL'].plot(ax=ax2)
    results[['AAPL_SMA', 'AAPL_LMA']].plot(ax=ax2)

    perf_trans = results.ix[[t != [] for t in results.transactions]]
    buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
    sells = perf_trans.ix[
        [t[0]['amount'] < 0 for t in perf_trans.transactions]]
    ax2.plot(buys.index, results['AAPL_SMA'].ix[buys.index],
             '^', markersize=10, color='m')
    ax2.plot(sells.index, results['AAPL_SMA'].ix[sells.index],
             'v', markersize=10, color='k')
    ax2.set_ylabel('price in $')
    plt.legend(loc=0)
    plt.show()


def main():
    # Set tz to avoid TypeError: Cannot compare tz-naive and tz-aware timestamps pandas issue
    # This shold be fixed in 0.23
    start = (pd.to_datetime('2018-01-01') + BDay(1)).tz_localize('US/Eastern')
    end = (pd.to_datetime('2018-08-28') + BDay(1)).tz_localize('US/Eastern')
    capital_base = 100000.0

    data = load_from(ASSETS)
    backtest = run_algorithm(start, end,
                             initialize, capital_base,
                             handle_data=handle_data,
                             analyze=analyze,
                             data=data)
    # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(backtest)
    # pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')
    # plt.show()


if __name__ == '__main__':
    main()
