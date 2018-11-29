"""
Download options data from http://www.cboe.com/delayedquote/quote-table
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import cm
import pandas_datareader as pdr

from py_vollib import black_scholes_merton as BSM
from py_vollib.black_scholes_merton import implied_volatility as IV
from py_vollib.black_scholes.greeks.analytical import vega
from py_lets_be_rational.exceptions import VolatilityValueException


RISK_FREE_INTEREST_RATE = 0.0225


class ImpliedVolatilityHelper(object):

    @classmethod
    def calculate(cls, S, K, t, V_market, r=RISK_FREE_INTEREST_RATE, q=0, flag='c'):
        try:
            return IV.implied_volatility(V_market, S, K, t, r, q, flag)
        except VolatilityValueException:
            return 0.0

    @classmethod
    def read_stock_data(cls, symbols, start_date, end_date, provider='yahoo', benchmark_symbol='SPY', fetch_benchmark=False):
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

    @classmethod
    def calculate_returns(cls, df, window=30, close='Close'):
        # Compute liner return (KDE, hist, etc)
        df['LinerReturn'] = df[close].pct_change(1)
        # Compute the logarithmic returns using the Closing price
        df['LogReturn'] = np.log(
            df[close] / df[close].shift(1))
        # Compute Volatility using the pandas rolling standard deviation function
        df['Volatility'] = df['LogReturn'].rolling(
            window=window).std() * np.sqrt(252)
        return df

    @classmethod
    def plot(cls, options, matplotlib_style, historic_volatility=False, output_image=None):
        print('Calculating Implied Volatility for Calls...')
        call_iv = pd.Series(pd.np.zeros(len(options.data.index)),
                            index=options.data.index,
                            name='_CallImpliedVolatility')
        for i in options.data.index:
            t = options.data.Expiration[i] / 365.0
            price = (options.data.CallBid[i] + options.data.CallAsk[i]) / 2.0
            call_iv[i] = cls.calculate(options.underlying_price,
                                       options.data.Strike[i],
                                       t,
                                       price,
                                       flag='c')

        print('Calculated Implied Volatility for %d Calls' % len(options.data.index))
        data = options.data.join(call_iv)

        print('Calculating Implied Volatility for Puts...')
        put_iv = pd.Series(np.zeros(len(options.data.index)),
                           index=options.data.index,
                           name='_PutImpliedVolatility')

        for i in options.data.index:
            t = options.data.Expiration[i] / 365.0
            price = (options.data.PutBid[i] + options.data.PutAsk[i]) / 2.0
            put_iv[i] = cls.calculate(options.underlying_price,
                                      options.data.Strike[i],
                                      t,
                                      price,
                                      flag='p')

        print('Calculated Implied Volatility for %i Puts' % len(options.data.index))
        data = data.join(put_iv)

        mpl.style.use(matplotlib_style)
        fig = plt.figure(1, figsize=(16, 12))
        # fig.subplots_adjust(hspace=.6, wspace=.2)

        # Plot the Implied Volatility curves
        index = 0
        nrows = ncols = math.ceil(np.sqrt(len(options.plot_dates_dict)))
        # Add 2 extra rows for the historic volatility and price graphs
        if historic_volatility:
            nrows += 2
        # Make sure we have at least 4 columns
        ncols = ncols if ncols >= 4 else 4

        max_xticks = 5

        def _enforce_fontsize(ax, fontsize=9):
            # Set font size for all IV chart elements
            for item in ([ax.title,
                          ax.xaxis.label,
                          ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(9)

        def _emphasise_text(text, style='bf'):
            """
            Supported styles are bf for Bold and it for Italics
            See https://matplotlib.org/users/mathtext.html for more info
            """
            return ' '.join(['$\\' + style + '{' + word + '}$' for word in text.split()])

        for key, value in options.plot_dates_dict.items():
            index += 1
            expiration = (value - options.current_time).days
            plot_call = data[(data._CallImpliedVolatility > .01) &
                           (data._CallImpliedVolatility < 1) &
                           (data.Expiration == expiration) &
                           (data.CallLastSale > 0)]
            plot_put = data[(data._PutImpliedVolatility > .01) &
                            (data._PutImpliedVolatility < 1) &
                            (data.Expiration == expiration) &
                            (data.PutLastSale > 0)]

            row = math.floor((index - 1) / ncols)
            ax1 = plt.subplot2grid((nrows, ncols), (row, (index - 1) % ncols), colspan=1, rowspan=1, fig=fig)

            xloc = plt.MaxNLocator(max_xticks)
            ax1.xaxis.set_major_locator(xloc)
            ax1.set_title('{} @ {} - {}'.format(options.symbol, options.underlying_price, _emphasise_text(key)))

            ax1.plot(plot_call.Strike, plot_call._CallImpliedVolatility,
                         marker='o', color='red',
                         markersize=3, linewidth=0, label='call')
            ax1.plot(plot_put.Strike, plot_put._PutImpliedVolatility,
                         marker='o', color='blue',
                         markersize=3, linewidth=0, label='put')

            ax1.legend(loc=1, numpoints=1, frameon=True, prop={'size': 10})
            ax1.set_ylim([0, 1])
            ax1.set_xlabel('Strike Price')
            ax1.set_ylabel('IV')
            ax1.margins(x=0)

            _enforce_fontsize(ax1, fontsize=9)


        if historic_volatility:
            delta = timedelta(days=365)
            stocks = cls.read_stock_data(
                options.symbol,
                start_date=(options.current_time - delta),
                end_date=options.current_time)
            stock = stocks[options.symbol]
            cls.calculate_returns(stock)

            years = plt.matplotlib.dates.YearLocator()
            months = plt.matplotlib.dates.MonthLocator()
            weekday = plt.matplotlib.dates.WeekdayLocator()
            formatter = plt.matplotlib.dates.DateFormatter('%m-%y')

            row = math.floor((index - 1) / ncols) + 1
            colspan = math.floor(ncols / 2)
            ax1 = plt.subplot2grid((nrows, ncols), (row, 0), colspan=colspan, rowspan=2, fig=fig)
            ax1.set_title('Historical Volatility', fontsize=10)
            ax1.plot(stock['Volatility'], linestyle='-', linewidth=1)
            ax1.set_ylim([0, 1])
            ax1.grid(True)
            ax1.xaxis.set_major_locator(months)
            ax1.xaxis.set_major_formatter(formatter)
            ax1.xaxis.set_minor_locator(weekday)
            ax1.margins(x=0)
            _enforce_fontsize(ax1, fontsize=10)

            ax1 = plt.subplot2grid((nrows, ncols), (row, colspan), colspan=colspan, rowspan=2, fig=fig)
            ax1.set_title('%s - %s @ %s for %s' % (options.symbol, options.company,
                              options.underlying_price, options.current_time), fontsize=10)
            ax1.plot(stock['Close'], linestyle='-', linewidth=1)
            ax1.grid(True)
            ax1.xaxis.set_major_locator(months)
            ax1.xaxis.set_major_formatter(formatter)
            ax1.xaxis.set_minor_locator(weekday)
            ax1.margins(x=0)
            _enforce_fontsize(ax1, fontsize=10)

            # ax1 = plt.subplot2grid((nrows, ncols), (row + 2, 0), colspan=2, rowspan=2, fig=fig)
            # # KDE (Kernel Density Estimation)
            # stock['LinerReturn'].plot(label='KDE', ax=ax1, kind='kde')
            # # stock['LinerReturn'].plot(label='Histogram', ax=ax1, kind='hist', alpha=0.6, bins=100, secondary_y=True)
            # ax1.grid(True)

            # mean = stock['LinerReturn'].mean()
            # ax1.annotate('mean', xy=(mean, 0.008), xytext=(mean + 10, 0.010))
            # # # vertical dotted line originating at mean value
            # ax1.axvline(mean, linestyle='--', linewidth=2, color='red')

            # ax1.margins(x=0)
            # _enforce_fontsize(ax1, fontsize=10)

        fig.tight_layout()

        if output_image:
            fig.savefig(cls.normalize_path(output_image), bbox_inches='tight')
        else:
            plt.show()

    @classmethod
    def weighted_iv(cls, options, term, flag='c'):
        if flag == 'c':
            # Call fields
            bid = 'CallBid'
            ask = 'CallAsk'
            last_sale = 'CallLastSale'
        else:
            # Put fields
            bid = 'PutBid'
            ask = 'PutAsk'
            last_sale = 'PutLastSale'

        option_iv = pd.Series(pd.np.zeros(len(options.data.index)),
                            index=options.data.index,
                            name='OptionImpliedVolatility')
        for i in options.data.index:
            t = options.data.Expiration[i] / 365.0
            price = (options.data[bid][i] + options.data[ask][i]) / 2.0
            option_iv[i] = cls.calculate(options.underlying_price,
                                         options.data.Strike[i],
                                         t,
                                         price,
                                         flag=flag)
        data = options.data.join(option_iv)

        def _find_expiration_dates(df, term):
            """
            Find two expiration dates closest to given number of days, front and back month.
            """
            expirations = df[['Expiration']].drop_duplicates().reset_index(drop=True)
            back_expirations = expirations[(expirations.Expiration > term)].copy()
            front_expirations = expirations[(expirations.Expiration < term)].copy()
            if back_expirations.size > 0 and front_expirations.size > 0:
                front_back_values = []
                for frame in [front_expirations, back_expirations,]:
                    frame['Difference'] = frame['Expiration'].apply(lambda value: abs(value - term))
                    front_back_values.append(frame.sort_values(by=['Difference']).head(1)['Expiration'].values[0])
                return front_back_values
            return None, None

        front, back = _find_expiration_dates(data, term)
        if not front or not back:
            return 0.0

        def _find_closest_strikes(df, front, back, underlying_price, number_of_strikes=4):
            """
            We take 4 (number_of_strikes) options contracts with strikes
            nearest to current stock price. We'll use this list to
            calculate IV Index for the front and back months.
            """
            strike_expiration_df = df[
                (df.Expiration == front) | (df.Expiration == back)
            ][['Strike', 'Expiration']].drop_duplicates().reset_index(drop=True)
            front_back_lists = []
            for value in (front, back,):
                s1 = strike_expiration_df[(strike_expiration_df.Expiration == value)]['Strike'].apply(
                    lambda item: abs(item - underlying_price)).reset_index(drop=True)
                s2 = df[(df.Expiration == value)]['Strike'].drop_duplicates().reset_index(drop=True)
                front_back_lists.append(pd.DataFrame({'Difference': s1, 'Strike': s2}).reset_index(drop=True).sort_values(
                    by='Difference').head(number_of_strikes)['Strike'].values)
            return front_back_lists
        
        front_strikes, back_strikes = _find_closest_strikes(data, front, back, options.underlying_price)
        # data = data[(data.OptionImpliedVolatility > .01) & (data.OptionImpliedVolatility < 1) &
        # TODO: Removed IV < 1 filter. IV can be greater than 100%. Looking at you AyyMD ;)
        data = data[(data.OptionImpliedVolatility > .01) &
                    (
                        ((data.Expiration == front) & (data.Strike.isin(front_strikes))) |
                        ((data.Expiration == back) & (data.Strike.isin(back_strikes)))
                    ) & ((data[bid] > 0) | (data[ask] > 0))]
                    # ) & (data[last_sale] > 0) & (data[bid] > 0) & (data[ask] > 0)]
                    # TODO: Do we care about filtering out options that are not active (last_sale == 0). For example,
                    #       if we keep only last_sale > 0 strikes we may end up without options to process. This will
                    #       cause vega weights calculations to fail.
                    # TODO: Sometimes bid or ask are 0, so make sure average is calculated correctly.

        df = pd.DataFrame()
        for index, items in data[['Strike', 'Expiration']].drop_duplicates().reset_index(drop=True).iterrows():
            values = data[(data.Strike == items['Strike']) & (data.Expiration == items['Expiration'])][['OptionImpliedVolatility', ask, bid]].mean()
            values['Strike'] = items['Strike']
            values['Expiration'] = items['Expiration']
            values['Price'] = (values[bid] + values[ask]) / 2.0 if values[ask] > 0 and values[bid] > 0 else max(values[ask], values[bid])
            values['Vega'] = vega(flag,
                                  options.underlying_price,
                                  items['Strike'],
                                  items['Expiration'] / 365.0,
                                  RISK_FREE_INTEREST_RATE,
                                  values['OptionImpliedVolatility'])
            df = df.append(values, ignore_index=True)

        # Liquidity considerations in estimating implied volatility (Rohini Grover and Susan Thomas)
        #
        # Interpolation scheme:
        #
        # vix = 100 * [ weight_t1 * ( (Nc2 - 30) / (Nc2 - Nc1) ) + weight_t2 * ( (30 - Nc1) / (Nc2 - Nc1) ) ]
        #
        # Where weight_ti are implied volatilities and Nci is the number of calendar days to
        # expiration. Here, i = 1, 2 for the near and next month respectively.

        def _front_vega_weights(df, front, back, term):
            """
            This function will calculate weighted average for the front month,
            where weighting is done by Vega (option price sensitivity to a
            change in Implied Volatility).
            """
            vegas_ivs = 0.0
            vegas = 0.0
            for index, items in df[(df.Expiration == front)].iterrows():
                vegas_ivs += items['Vega'] * items['OptionImpliedVolatility']
                vegas += items['Vega']
            vw = vegas_ivs / vegas
            expiry_sqrt = (math.sqrt(back) - math.sqrt(term)) / (math.sqrt(back) - math.sqrt(front))
            return vw, expiry_sqrt

        def _back_vega_weights(df, front, back, term):
            """
            This function will calculate weighted average for the back month,
            where weighting is done by Vega (option price sensitivity to a
            change in Implied Volatility).
            """
            vegas_ivs = 0.0
            vegas = 0.0
            for index, items in df[(df.Expiration == back)].iterrows():
                vegas_ivs += items['Vega'] * items['OptionImpliedVolatility']
                vegas += items['Vega']
            vw = vegas_ivs / vegas
            expiry_sqrt = (math.sqrt(term) - math.sqrt(front)) / (math.sqrt(back) - math.sqrt(front))
            return vw, expiry_sqrt

        vw1, i1 = _front_vega_weights(df, front, back, term)
        vw2, i2 = _back_vega_weights(df, front, back, term)

        return i1 * vw1 + i2 * vw2
