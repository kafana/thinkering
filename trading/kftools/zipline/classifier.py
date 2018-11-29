import os
import sys
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import deque
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pylab as plt
import pyfolio as pf
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    set_benchmark,
    set_commission,
    set_slippage,
    symbol,
    symbols,
)
from zipline.data.benchmarks import get_benchmark_returns
from zipline.utils.run_algo import run_algorithm
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import SimpleMovingAverage, RSI, \
                                     MovingAverageConvergenceDivergenceSignal, \
                                     ExponentialWeightedMovingAverage, \
                                     BollingerBands, Returns

from kftools.optimizations import PortfolioOptimizer
from kftools import plotting

working_dir = os.getcwd()
working_home_dir = os.path.basename(working_dir)
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pickle_file = 'zipline-results.pickle'
export_file_format = '{}-tear-sheet.png'


ASSETS = ['IBM', 'SBUX', 'XOM', 'AAPL', 'MSFT']
# ASSETS = ['AAPL', 'MSFT']
ADDITIONAL_ASSETS = ['QQQ', 'SPY']
ALL_ASSETS = ASSETS + ADDITIONAL_ASSETS


def make_pipeline():
    all_assets_filter = (USEquityPricing.close.latest > 0)
    returns_5 = Returns(window_length=5)
    rsi = RSI(inputs=[USEquityPricing.close])
    macd = MovingAverageConvergenceDivergenceSignal(
        mask=all_assets_filter
    )
    ema = ExponentialWeightedMovingAverage(
        mask=all_assets_filter,
        inputs=[USEquityPricing.close],
        window_length=30,
        decay_rate=(1 - (2.0 / (1 + 15.0)))
    )
    mean_5 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=5,
        mask=all_assets_filter
    )
    mean_10 = SimpleMovingAverage(
        inputs=[USEquityPricing.close],
        window_length=10,
        mask=all_assets_filter
    )
    bb = BollingerBands(
        inputs=[USEquityPricing.close],
        window_length=20,
        k=2
    )
    return Pipeline(
        columns={
            'returns_5': returns_5,
            'RSI': rsi,
            'MACD': macd,
            'EMA': ema,
            'SMA_5': mean_5,
            'SMA_10': mean_10,
            'bb_upper': bb.upper,
            'bb_middle': bb.middle,
            'bb_lower': bb.lower
        },
        screen=all_assets_filter,
    )


def initialize(context):
    attach_pipeline(make_pipeline(), 'data_pipeline')
    
    context.technical_indicator_states = {}
    context.window_length = 7
    context.benchmark = deque(maxlen=context.window_length)
    context.benchmark_asset = symbol('SPY')
    context.benchmark_assets = symbols('QQQ', 'SPY')

    #context.classifier = RandomForestClassifier() # Use a random forest classifier
    context.classifier = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10)

    # Clasifier training data
    context.features = ['RSI','EMA','MACD','SMA_5','SMA_10','bb_lower','bb_middle','bb_upper']
    context.response = ['Class']
    context.X = pd.DataFrame(columns=context.features) # Independent, or input variables
    context.Y = pd.DataFrame(columns=context.response) # Dependent, or output variable

    context.prediction = {} # Stores most recent prediction
    
    context.tick = 0
    context.total_buy = 0
    context.positions = None
    context.position_adjustment_days = 5 # Number of days to wait before adjusting positions
    context.min_data_points = 500
    context.max_data_points = 1500

    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open(minutes=1))
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
    set_benchmark(symbol('SPY'))
    # Turn off the slippage model
    set_slippage(slippage.FixedSlippage(spread=0.0))
    # Set the commission model (Interactive Brokers Commission)
    set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))


def record_vars(context, data):
    record(total_buy=context.total_buy)


def before_trading_start(context, data):
    context.all_assets = pipeline_output('data_pipeline')
    context.output = context.all_assets
    # context.output = context.output.rename_axis(['Name'])


def _is_benchamrk_ready(context):
    return True if len(context.benchmark) == context.benchmark.maxlen else False

def _is_equity_ready(context, symbol):
    return True if len(context.technical_indicator_states[symbol].index) == context.window_length else False

def _adjust_position(context):
    return True if context.tick % context.position_adjustment_days == 0 else False

def _add_to_portfolio(prediction_array):
    # Note: We should be able to run prediction_array > 0 condition
    return True if prediction_array.size == 1 and prediction_array[0] else False

def rebalance(context, data):
    context.tick += 1

    for equity, _ in context.output.iterrows():
        if equity in context.technical_indicator_states:
            context.technical_indicator_states[equity] = context.technical_indicator_states[equity].append(
                context.output.loc[equity], ignore_index=True)
        else:
            context.technical_indicator_states[equity] = pd.DataFrame(
                context.output.loc[equity]).transpose()

    context.benchmark.append(data.current(context.benchmark_asset, 'price'))

    # Wait till we accumulate enough data inside of the benchmark collection
    if _is_benchamrk_ready(context):

        # Calculate Benchmark return
        benchmark = (context.benchmark[-1] - context.benchmark[0]) / context.benchmark[0]
        symbol_X_tests = {}

        for equity, _ in context.output.iterrows():
            # Make sure there is enough data in for the equity
            if _is_equity_ready(context, equity):
                # Take the last return and check if it beat benchmark
                returns_5 = context.technical_indicator_states[equity].iloc[-1]['returns_5']
                change = returns_5 > benchmark and returns_5 > 0

                Y_train = {}
                Y_train['Class'] = change

                # Load X train data from the 1st row
                X_train = {}
                for column in context.technical_indicator_states[equity].columns:
                    if column in context.features:
                        X_train[column] = context.technical_indicator_states[equity].iloc[0][column]

                context.X = context.X.append([X_train], ignore_index=True)
                context.Y = context.Y.append([Y_train], ignore_index=True)

                # Load X test data from the latest row
                symbol_test = {}
                for column in context.technical_indicator_states[equity].columns:
                    if column in context.features:
                        symbol_test[column] = context.technical_indicator_states[equity].iloc[-1][column]

                symbol_X_tests[equity] = symbol_test

                # Purge 1st row
                context.technical_indicator_states[equity] = context.technical_indicator_states[equity].iloc[1:]

        # There needs to be enough data points to make a good model and adjust positions once per 5 days
        if len(context.X.index) >= context.min_data_points and _adjust_position(context):

            # Purge data if we reach max number of endpoints (do we have to to this?)
            if len(context.X.index) > context.max_data_points:
                context.X = context.X.iloc[context.min_data_points:]
                context.Y = context.Y.iloc[context.min_data_points:]

            context.Y[context.response] = context.Y[context.response].astype('bool')
            context.classifier.fit(context.X[context.features].values, context.Y[context.response].values) # Generate the model

            context.total_buy = 0
            for equity, _ in context.output.iterrows():
                if equity in symbol_X_tests:
                    X_test = pd.DataFrame(columns=context.features)
                    X_test = X_test.append(symbol_X_tests[equity], ignore_index=True)
                    context.prediction[equity] = context.classifier.predict(X_test)
                    context.total_buy += 1 if context.prediction[equity][0] else 0

            position_symbols = []
            for equity, _ in context.output.iterrows():
                if equity in context.prediction:
                    if _add_to_portfolio(context.prediction[equity]):
                        position_symbols.append(equity)
                    else:
                        order_target_percent(equity, 0.0)

            if context.total_buy != 0:
                context.positions = data.history(position_symbols, 'close', 30, '1d')
                weights = PortfolioOptimizer.optimize_weights1(context.positions)
                for equity in weights.keys():
                    order_target_percent(equity, weights[equity])


def analyze(context, backtest):
    # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(backtest)
    # ax = pf.plot_drawdown_periods(returns, top=5)
    # ax.set_xlabel('Date')
    # plt.show()
    pass


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(prog='zipline-classifier',
                                     description='Zipline classifier')
    parser.add_argument('-s', '--symbols', nargs='+', type=str, help='Stock symbols to export')
    parser.add_argument('-f', '--from-date', type=str, help='Start date in %%Y-%%m-%%d format')
    parser.add_argument('-t', '--to-date', type=str, help='End date in %%Y-%%m-%%d format')
    parser.add_argument('-c', '--capital-base', type=float, help='Capital base, default to 100000.0', default=100000.0)

    return parser.parse_args(args=args)

def main():
    args = parse_arguments()
    # Set tz to avoid TypeError: Cannot compare tz-naive and tz-aware timestamps pandas issue
    # This shold be fixed in 0.23
    start = (pd.to_datetime('2017-08-01') + BDay(1)).tz_localize('UTC')
    end = (pd.to_datetime('2018-08-28') + BDay(1)).tz_localize('UTC')

    # TODO: Add start/end date setup
    # start = pd.Timestamp(args.from_date, tz='utc') if args.from_date else None
    # end = pd.Timestamp(args.to_date, tz='utc') if args.to_date else None

    backtest = run_algorithm(start, end,
                             initialize,
                             args.capital_base,
                             before_trading_start=before_trading_start,
                             analyze=analyze,
                             bundle='iex-csvdir-bundle')
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(backtest)
    # TODO: The backtest benchmark graphs/data are not looking good, check why is this happening.
    benchmark_rets = None # backtest.benchmark_period_return
    plotting.create_simple_tear_sheet(returns,
                                      positions=positions,
                                      transactions=transactions,
                                      benchmark_rets=benchmark_rets,
                                      file_name=os.path.join(working_dir, export_file_format.format('simple')))

    plotting.create_returns_tear_sheet(returns,
                                       positions=positions,
                                       transactions=transactions,
                                       benchmark_rets=benchmark_rets,
                                       file_name=os.path.join(working_dir, export_file_format.format('returns')))


if __name__ == '__main__':
    main()
