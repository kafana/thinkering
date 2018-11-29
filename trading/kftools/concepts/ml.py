#!/usr/bin/env python

"""
Misc ML examples and prototyping
"""
import sys
import subprocess
from datetime import datetime
from functools import partial

import pandas as pd
import pandas_datareader as pdr
import mibian as mb
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # Sample data
import matplotlib.pyplot as plt
import numpy as np
import graphviz

def get_code(tree, feature_names, target_names, spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)


def read_stock_data(symbols, start_date, end_date, provider='yahoo', benchmark_symbol='SPY', fetch_benchmark=False):
    stocks = {}
    if not isinstance(symbols, (list, tuple,)):
        symbols = [symbols]
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    for symbol in symbols:
        stocks[symbol] = pdr.DataReader(symbol, provider,
                                        start_date, end_date)

    if fetch_benchmark and benchmark_symbol:
        stocks[benchmark_symbol] = pdr.DataReader(benchmark_symbol, provider,
                                                  start_date, end_date)

    return stocks

def visualize_tree(tree, feature_names=None, target_names=None):
    dot_data = export_graphviz(tree,
                               out_file=None,
                               feature_names=feature_names,
                               class_names=target_names,  
                               filled=True,
                               rounded=True,  
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("options-graph")

def plot_dt():
    # iris = load_iris()
    # print(iris.target_names)
    rate = 0.01
    strike_price = 315
    start_date=pd.to_datetime('2018-01-01')
    end_date=pd.to_datetime('2018-08-25')

    stocks = read_stock_data('SPY', start_date, end_date)
    options_data = pd.read_csv("spy-2018-09-28.csv", index_col='Date')

    index_data = stocks['SPY']
    index_data.reset_index(inplace=True)
    index_data['Date'] = index_data['Date'].apply(lambda item: pd.to_datetime(item.strftime('%Y-%m-%d')))
    index_data.set_index('Date', inplace=True)

    options_data.reset_index(inplace=True)
    options_data['Date'] = options_data['Date'].apply(lambda item: pd.to_datetime(item))
    options_data.set_index('Date', inplace=True)

    options_data = options_data[
        (options_data.StrikePrice == strike_price) &
        (options_data.PutCall == 'call')]
    options_data['OptionPrice'] = (options_data['AskPrice'] + options_data['BidPrice']) / 2.0

    index_data = index_data[['Adj Close']]
    index_data = index_data.rename(columns={'Adj Close': 'IndexPrice'})
    # 'ImpliedVolatility', 'Delta', 'Gamma', 'Vega'
    options_data = options_data[['OptionPrice', 'DaysToExpiration', 'ImpliedVolatility', 'Delta', 'Gamma', 'Vega']]
    options_data['Vega'] = options_data['Vega'] / 100.0
    options_data = options_data.rename(columns={'ImpliedVolatility': 'IV'})

    merged_data = pd.merge(index_data, options_data, how='inner', left_index=True, right_index=True)
    
    merged_data['IV.1'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price,
        rate, item['DaysToExpiration']],
        callPrice=item['OptionPrice']).impliedVolatility, axis=1)
    merged_data['Delta.1'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price,
        rate, item['DaysToExpiration']],
        volatility=item['IV.1']).callDelta, axis=1)
    merged_data['Vega.1'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price, rate,
        item['DaysToExpiration']],
        volatility=item['IV.1']).vega, axis=1)
    merged_data['Gamma.1'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price, rate,
        item['DaysToExpiration']],
        volatility=item['IV.1']).gamma, axis=1)
    merged_data['Theta.1'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price, rate,
        item['DaysToExpiration']],
        volatility=item['IV.1']).callTheta, axis=1)

    # Discount Options Data is missing Theta
    merged_data['Theta'] = merged_data.apply(lambda item: mb.BS(
        [item['IndexPrice'], strike_price, rate,
        item['DaysToExpiration']],
        volatility=item['IV'] * 100.0).callTheta, axis=1)

    # Create signal for the tree fit Y param
    merged_data['Signal'] = 0
    merged_data.loc[merged_data['OptionPrice'].shift(-1) < merged_data['OptionPrice'], 'Signal'] = -1
    merged_data = merged_data.dropna()

    # x = merged_data[['IV.1', 'Delta.1', 'Vega.1', 'Gamma.1', 'Theta.1']]
    # feature_names = ['IV', 'Delta', 'Vega', 'Gamma', 'Theta']
    feature_names = ['IV.1', 'Delta.1', 'Vega.1', 'Gamma.1', 'Theta.1']
    x = merged_data[feature_names]
    target_names = ['Signal']
    y = merged_data['Signal']

    # i = int(0.85 * len(x))
    # x_train, x_test, y_train, y_test = x[:i], x[i:], y[:i], y[i:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.15, train_size=0.85)

    dtc = DecisionTreeClassifier(random_state=1001)
    dtc = dtc.fit(x_train, y_train)

    merged_data['Predicted'] = dtc.predict(x)
    merged_data['DailyReturn'] = merged_data['OptionPrice'].shift(-1) - merged_data['OptionPrice']
    merged_data['Profit'] = merged_data['Predicted'] * merged_data['DailyReturn']

    visualize_tree(dtc, feature_names=feature_names, target_names=None)

    accuracy = accuracy_score(y_test, dtc.predict(x_test))
    profit = np.nansum(merged_data['Profit'].loc[x_test.index].values)
    plt.plot(merged_data['Profit'].loc[x_test.index].values.cumsum())

    print("Accuracy: ", accuracy, "Profit: ", profit)

    plt.show()

def main():
    plot_dt()

if __name__ == "__main__":
    main()
