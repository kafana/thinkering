import os
import sys
import enum
import copy
import math
import glob
import logging
import argparse
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial


from dateutil import parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as plticker
from scipy.stats import norm
from scipy.signal import argrelextrema
from py_vollib import black_scholes_merton as BSM
from py_vollib.black_scholes_merton import implied_volatility as IV
from py_lets_be_rational.exceptions import VolatilityValueException
from iexfinance import StockReader


from kftools.iv import ImpliedVolatilityHelper
from kftools.models import Manager, DatabaseClient, ImpliedVolatility


logger = logging.getLogger(__name__)

working_dir = os.getcwd()


implied_volatility_terms = {
    'iv30': 30,
    'iv60': 60,
    'iv90': 90,
    'iv120': 120,
}


call_month_codes = {
    'A': 'January',
    'B': 'February',
    'C': 'March',
    'D': 'April',
    'E': 'May',
    'F': 'June',
    'G': 'July',
    'H': 'August',
    'I': 'September',
    'J': 'October',
    'K': 'November',
    'L': 'December'
}


put_month_codes = {
    'M': 'January',
    'N': 'February',
    'O': 'March',
    'P': 'April',
    'Q': 'May',
    'R': 'June',
    'S': 'July',
    'T': 'August',
    'U': 'September',
    'V': 'October',
    'W': 'November',
    'X': 'December'
}


# http://www.cboe.com/delayedquote/help#chaindata
# This letter (if present) indicates which exchange(s) the option trades at, whereby:
# No hyphen or letter present = Composite
cboe_exchanges = {
    None: 'Composite',
    'A': 'AMEX American Stock Exchange',
    'B': 'BOX Boston Stock Exchange - Options',
    'E': 'Cboe Options Exchange',
    'I': 'BATS',
    'J': 'NASDAQ OMX BX',
    'O': 'NASDAQ OMX',
    'P': 'NYSE Arca',
    'X': 'PHLX Philadelphia Stock Exchange',
    'Y': 'C2 Exchange',
    '4': 'Miami Options Exchange',
    '8': 'ISE International Securities Exchange'
}


OptionsData = namedtuple('Options', 'data plot_dates_dict symbol underlying_price company '
                         'day_count current_time call_open_interest put_open_interest')
OptionCode = namedtuple('OptionCode', 'option_type symbol expiration '
                        'mini_option strike exchange code default')

def CND(d):
    """
    Cumulative Standard Normal Distribution
    """
    (a1, a2, a3, a4, a5) = (0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429)
    L = np.abs(d)
    K = 1.0 / (1.0 + 0.2316419 * L)
    w = 1.0 - 1.0 / np.sqrt(2 * np.pi) * np.exp(-L * L / 2.) * (
        a1 * K + a2 * K * K + a3 * (K ** 3) + a4 * (K ** 4) + a5 * (K ** 5))
    if d < 0:
        w = 1.0 - w
    return w


def probability_option_strategist(underlying_price, strike_price, expiration, volatility):
    """
    http://www.optionstrategist.com/calculators/probability
    """
    t = expiration / 365.0
    d1 = np.log(strike_price / underlying_price) / (volatility * np.sqrt(t))

    # Calculate Cumulative Standard Normal Distribution
    y = np.floor(1 / (1 + .2316419 * np.abs(d1)) * 100000) / 100000
    z = np.floor(.3989423 * np.exp(-((d1 * d1) / 2)) * 100000) / 100000

    y5 = 1.330274 * (y ** 5)
    y4 = 1.821256 * (y ** 4)
    y3 = 1.781478 * (y ** 3)
    y2 = .356538 * (y ** 2)
    y1 = .3193815 * y
    x = 1.0 - z * (y5 - y4 + y3 - y2 + y1)
    x = np.floor(x * 100000) / 100000

    if d1 < 0:
        x = 1.0 - x

    prob_above_strike = np.floor(x * 1000) / 10
    prob_below_strike = np.floor((1 - x) * 1000) / 10

    return prob_above_strike, prob_below_strike


def probability_in_money(underlying_price, strike_price, expiration, volatility, flag='c'):
    """
    Probability of a successful trade for:

    * European put finishing in the money (that is, the probability that the
      strike price is above the market price at maturity).
    * European call finishing in the money (that is, the probability that
      the strike price is below the market price at maturity)

    PUT = 1 - N( ln(S / K) / IV * SQRT(t) )
    CALL = N( ln(S / K) / IV * SQRT(t) )

    N = normal distribution
    S = underlying stock price
    K = strike price
    IV = implied volatility
    t = time to expiry

    These equations are closely related to the Delta of an option. We can use
    delta as approximation of the likelihood of an option finishing in the money.

    DELTA = N( (ln(S / X) + (r + (IV ^ 2 / 2)) * t) / (IV * SQRT(t)) )
    """
    t = expiration / 365.
    nlsou = np.log(strike_price / underlying_price)
    if flag == 'c':
        above = 1 - norm.cdf(nlsou / (volatility * np.sqrt(t)))
        win = norm.cdf(nlsou / (volatility * np.sqrt(t)))
        return win, above
    else:
        win = 1 - norm.cdf(nlsou / (volatility * np.sqrt(t)))
        below =  norm.cdf(nlsou / (volatility * np.sqrt(t)))
        return win, below

def put_probability(winning_prob, underlying_price, expiration, volatility):
    """
    This function calculates the desired lower stock price for a
    winning trade, given a probability.
    """
    t = expiration / 365.
    price = (np.e ** (
        (volatility / 100.) * np.sqrt(t) * -1 * norm.ppf(winning_prob)
    )) * underlying_price
    return price

def call_probability(winning_prob, underlying_price, expiration, volatility):
    """
    This function calculates the desired higher stock price for a
    winning trade, given a probability.
    """
    t = expiration / 365.
    price = (np.e ** (
        (volatility / 100.) * np.sqrt(t) * norm.ppf(winning_prob)
    )) * underlying_price
    return price

class OptionDataInputTypes(enum.Enum):
    cboe = 'cboe'
    discountoptions = 'discountoptions'

    def __str__(self):
        return self.value


class OptionPosition(enum.Enum):
    long = 'long'
    short = 'short'
    

class Option(ABC):
    def __init__(self, strike_price, expiration, bid_price, ask_price,
                 option_symbol=None, underlying_symbol=None,
                 position=OptionPosition.long,
                 volume=0, quantity=1):
        self.strike_price = strike_price
        self.expiration = expiration
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.volume = volume
        self.option_symbol = option_symbol
        self.quantity = quantity
        self.position = position

    @property
    def t(self):
        return self.expiration / 365.0

    @property
    def price(self):
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def contract_price(self):
        return (self.price if self.position == OptionPosition.long else self.price * -1.0) * self.quantity

    def _calculate_iv(self, underlying_price, q=0, flag='c', rate=0.02, expiration=None):
        expiration = self.t if expiration == None else expiration / 365.0
        try:
            return IV.implied_volatility(self.price, underlying_price,
                                         self.strike_price, expiration,
                                         rate, q, flag)
        except VolatilityValueException:
            return 0.0

    def _calculate_bsm(self, underlying_price, sigma, q=0, flag='c', rate=0.02, expiration=None):
        """
        Returns the Black-Scholes-Merton option price.

        underlying_price - underlying asset price
        sigma - annualized standard deviation, or volatility
        q - annualized continuous dividend rate
        """
        expiration = self.t if expiration == None else expiration / 365.0
        try:
            return BSM.black_scholes_merton(flag, underlying_price,
                                            self.strike_price, expiration,
                                            rate, sigma, q)
        except VolatilityValueException:
            return 0.0

    def __repr__(self):
        return "<Option(strike_price=%r, expiration=%r, position=%r)>" % (
            self.strike_price, self.expiration, self.position)

    @abstractmethod
    def iv(self, underlying_price, expiration=None):
        pass

    @abstractmethod
    def bsm(self, underlying_price, sigma, rate=None, expiration=None):
        pass

    @property
    @abstractmethod
    def label(self):
        pass

    @abstractmethod
    def payoff_range(self, strike_range):
        pass

    def premium_for_price_range(self, strike_range, iv=0.3, rate=0.02, days_from_now=0):
        r = np.array([self.bsm(price, iv,
                               rate=rate,
                               expiration=(self.expiration - days_from_now)) for price in strike_range]) * self.quantity
        return (r if self.position == OptionPosition.long else r * -1.0)


class PutOption(Option):
    @property
    def label(self):
        return '{} {:.2f} {} Put(s) @ ${:.2f}'.format(self.quantity,
            self.strike_price,
            self.position.value.capitalize(),
            self.price)

    def payoff_range(self, strike_range):
        r = np.where(strike_range < self.strike_price,
                     self.strike_price - strike_range, 0) - self.price
        r = r * self.quantity
        return (r if self.position == OptionPosition.long else r * -1.0)

    def iv(self, underlying_price, expiration=None):
        return self._calculate_iv(underlying_price, flag='p', expiration=expiration)

    def bsm(self, underlying_price, sigma, rate=None, expiration=None):
        return self._calculate_bsm(underlying_price, sigma, flag='p', rate=rate, expiration=expiration)


class CallOption(Option):
    @property
    def label(self):
        return '{} {:.2f} {} Call(s) @ ${:.2f}'.format(self.quantity,
            self.strike_price,
            self.position.value.capitalize(),
            self.price)

    def payoff_range(self, strike_range):
        r = np.where(strike_range > self.strike_price,
                     strike_range - self.strike_price, 0) - self.price
        r = r * self.quantity
        return (r if self.position == OptionPosition.long else r * -1.0)

    def iv(self, underlying_price, expiration=None):
        return self._calculate_iv(underlying_price, flag='c', expiration=expiration)

    def bsm(self, underlying_price, sigma, rate=None, expiration=None):
        return self._calculate_bsm(underlying_price, sigma, flag='c', rate=rate, expiration=expiration)


class OptionStrategyPlot(object):
    colors = ['red', 'green', 'yellow', 'orange', 'cyan', 'magenta']

    def __init__(self, underlying_price, options=None, strategy_label=None, plot_all=False):
        self.underlying_price = underlying_price
        self.options = options if isinstance(options, (list, tuple)) else [options]
        self.strategy_label = strategy_label
        self.plot_all = plot_all

    def _enforce_fontsize(self, ax, fontsize=9):
        # Set font size for all IV chart elements
        for item in ([ax.title,
                      ax.xaxis.label,
                      ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)

    def _create_plot_figure(self, figsize=(12, 9), matplotlib_style='seaborn'):
        plt.style.use(matplotlib_style)
        fig = plt.figure(1, figsize=figsize)
        return fig

    def _create_subplot(self, shape=(2, 2), loc=(0, 0), fig=None,
                        plots=None, title=None, text=None,
                        xlabel='Stock Price', ylabel='Profit & Loss', fontsize=9,
                        axis_linewidth=0.5, axis_color='#343434',
                        axis_tick_color=None, axis_label_color=None,
                        xaxis_locator_base=5.0, yaxis_locator_base=50.0):
        ax = plt.subplot2grid(shape, loc, colspan=1, rowspan=1, fig=fig)
        if title:
            ax.set_title(title)

        if text:
            anchored_text = AnchoredText(text, loc=2, pad=0.7, borderpad=0.7,
                                         frameon=True, prop={'fontsize': fontsize})
            ax.add_artist(anchored_text)

        ax.spines['top'].set_visible(False) # Remove top border
        ax.spines['right'].set_visible(False) # Remove right border
        ax.spines['bottom'].set_position('zero') # Center the X-axis

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(axis_linewidth)
            ax.spines[axis].set_color(axis_color)

        if axis_tick_color:
            ax.tick_params(axis='x', colors=axis_tick_color)
            ax.tick_params(axis='y', colors=axis_tick_color)

        if axis_label_color:
            ax.yaxis.label.set_color(axis_label_color)
            ax.xaxis.label.set_color(axis_label_color)

        for plot in plots:
            dc = copy.deepcopy(plot)
            args = dc.pop('args')
            ax.plot(*args, **dc)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=xaxis_locator_base))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=yaxis_locator_base))
        ax.legend(loc=1, frameon=True, fontsize=fontsize) # 0 = best

        self._enforce_fontsize(ax, fontsize=fontsize)

        return ax

    def _stock_price_range(self, underlying_price, range=(0.75, 1.25), force_step=None):
        """
        Creates option strike price range according to underlying stok price
        """
        if underlying_price < 10:
            step_or_loc_base = 1
        if 10 <= underlying_price < 50:
            step_or_loc_base = 2
        elif 50 <= underlying_price < 100:
            step_or_loc_base = 5
        elif 100 <= underlying_price < 500:
            step_or_loc_base = 10
        elif 500 <= underlying_price < 1000:
            step_or_loc_base = 20
        elif 1000 <= underlying_price < 2000:
            step_or_loc_base = 100
        elif 2000 <= underlying_price < 5000:
            step_or_loc_base = 250
        elif underlying_price >= 5000:
            step_or_loc_base = 500

        if force_step:
            step = force_step
        else:
            step = step_or_loc_base

        return step_or_loc_base, np.around(np.arange(
            range[0] * underlying_price, range[1] * underlying_price, step) / step, decimals=0
        ) * step

    @classmethod
    def _put_payoff(cls, strike_range, strike_price, price, is_short=False):
        r = np.where(strike_range < strike_price, strike_price - strike_range, 0) - price
        return (r * -1.0 if is_short else r) 

    @classmethod
    def _call_payoff(cls, strike_range, strike_price, price, is_short=False):
        r = np.where(strike_range > strike_price, strike_range - strike_price, 0) - price
        return (r * -1.0 if is_short else r) 

    def _calculate_payoff(self, payoffs):
        strategy_payoff = np.array([payoff for payoff in payoffs]).sum(axis=0)
        return strategy_payoff

    def _calculate_max_min_payoff(self, strategy_payoff):
        # Remove repeating values, and preserve order, so we can easily spot maxima and minima using argrelextrema
        # If this doesn't yield to any meaningful results we'll use min and max and check for repeating values
        _, indices = np.unique(strategy_payoff, return_index=True)
        unique = strategy_payoff[np.sort(indices)]
        maxima = argrelextrema(unique, np.greater) # Local maxima
        max_payoff = unique[maxima[0]][0] if len(maxima[0]) > 0 else np.PINF
        minima = argrelextrema(unique, np.less) # Local minima
        max_loss = unique[minima[0]][0] if len(minima[0]) > 0  else np.NINF
        # Lets see if we can get max value and min and detect repeating values
        if max_payoff == np.PINF:
            indices = np.argmax(strategy_payoff)
            if len(np.where(strategy_payoff == strategy_payoff[indices])[0]) > 1:
                max_payoff = strategy_payoff[indices]
        if max_loss == np.NINF:
            indices = np.argmin(strategy_payoff)
            if len(np.where(strategy_payoff == strategy_payoff[indices])[0]) > 1:
                max_loss = strategy_payoff[indices]
        return max_payoff, max_loss

    def _plot_strategy(self, additional_strategy_plots=[], plot_profit_loss_graph=True, iv=0.3, rate=0.02, days_from_now=0):
        shape = (1, 1) if not self.plot_all else (math.ceil(np.sqrt(len(self.options) + 2)),) * 2

        xaxis_locator_base, strike_range = self._stock_price_range(self.underlying_price, force_step=1)

        plots = []
        payoffs = []
        profit_loss_payoffs = []
        total_options_quantity = 0
        if len(self.options) > len(self.colors):
            colors = self.colors * math.ceil(len(self.options) / len(self.colors))
        else:
            colors = self.colors

        # Find the most recent strategy expiration time
        most_recent_expiration = min([option.expiration for option in self.options])

        total_contract_profit = 0.0
        for color, option in zip(colors, self.options):
            total_contract_profit += option.contract_price
            total_options_quantity += option.quantity
            sigma = option.iv(self.underlying_price, expiration=most_recent_expiration)
            option_payoff = option.premium_for_price_range(strike_range,
                                                           iv=sigma,
                                                           rate=rate,
                                                           days_from_now=most_recent_expiration)
            # Calculate different payoff array for graph (premium is included)
            option_payoff_with_premium = option.payoff_range(strike_range)
            option_payoff_plot = {'args': [strike_range, option_payoff_with_premium],
                                  'label': option.label,
                                  'color': color,
                                  'alpha': 0.6}
            payoffs.append(option_payoff)
            plots.append(option_payoff_plot)


            # #########################################################################################
            # Calculate values for the potential profit/loss graph
            if plot_profit_loss_graph:
                option_profit_loss_payoff = option.premium_for_price_range(strike_range,
                                                                           iv=iv,
                                                                           rate=rate,
                                                                           days_from_now=days_from_now)
                profit_loss_payoffs.append(option_profit_loss_payoff)
            # #########################################################################################

        strategy_payoff = self._calculate_payoff(payoffs) - total_contract_profit
        strategy_payoff_plot = {'args': [strike_range, strategy_payoff],
                                'label': '{} in {} day(s)'.format(
                                    self.strategy_label, most_recent_expiration),
                                'color': 'blue',
                                'linestyle': 'dashed',
                                'alpha': 0.8}
        plots.append(strategy_payoff_plot)

        # #############################################################################################
        # Add potential profit/loss graph
        if plot_profit_loss_graph:
            profit_loss_payoff = self._calculate_payoff(profit_loss_payoffs) - total_contract_profit
            profit_loss_payoff_plot = {'args': [strike_range, profit_loss_payoff],
                                                        'label': '{} in {} day(s) [iv={:.2f}, rate={:.2f}]'.format(
                                                            self.strategy_label, days_from_now, iv, rate),
                                                        'color': 'black',
                                                        'linestyle': '-.',
                                                        'alpha': 0.8}
            plots.append(profit_loss_payoff_plot)
        # #############################################################################################

        max_payoff, max_loss = self._calculate_max_min_payoff(strategy_payoff)

        index = 1
        loc = (math.floor((index - 1) / shape[1]), (index - 1) % shape[1])
        fig = self._create_plot_figure()
        yaxis_locator_base = xaxis_locator_base * math.floor(total_options_quantity / len(self.options))
        if self.plot_all:
            for plot in plots:
                ax = self._create_subplot(shape=shape, loc=loc, fig=fig, plots=[plot],
                                          xaxis_locator_base=xaxis_locator_base,
                                          yaxis_locator_base=yaxis_locator_base)
                index += 1
                loc = (math.floor((index - 1) / shape[1]), (index - 1) % shape[1])

        if additional_strategy_plots:
            plots = plots + additional_strategy_plots
        ax = self._create_subplot(shape=shape, loc=loc, fig=fig, plots=plots,
                                  text="Max Profit: {:.2f}\nMaximum Loss: {:.2f}".format(
                                      max_payoff, max_loss),
                                  xaxis_locator_base=xaxis_locator_base,
                                  yaxis_locator_base=yaxis_locator_base)
        fig.tight_layout()
        return fig

    def plot(self, **kwargs):
        self._plot_strategy(**kwargs)
        plt.show()

    def plot_to_file(self, file_name, **kwargs):
        fig = self._plot_strategy(**kwargs)
        fig.savefig(file_name)
        plt.close(fig)


class LongStraddlePlot(OptionStrategyPlot):
    """
    A long straddle is an options strategy where the trader purchases both a long call and a
    long put on the same underlying asset with the same expiration date and strike price.
    The strike price is at-the-money or as close to it as possible. The goal is to profit
    from a significant move in the underlying asset in either direction.
    """
    def __init__(self, underlying_price, long_put_option, long_call_option, plot_all=False):
        super(LongStraddlePlot, self).__init__(
            underlying_price,
            options=[long_put_option, long_call_option],
            strategy_label='Long Straddle',
            plot_all=plot_all)


class CollarPlot(OptionStrategyPlot):
    """
    Collar option strategies is a protective strategy that is implemented on a long stock
    position. An investor can create a collar position by purchasing an out-of-the-money put
    option while simultaneously writing an out-of-the-money call option. A collar is also
    known as hedge wrapper. The put protects the trader incase the price of the stock drops.
    Writing the call produces income (or offsets the cost of buying the put) and allows the
    trader to profit on the stock up to the strike price of the call, but not higher.
    """
    def __init__(self, underlying_price, long_put_option, short_call_option, plot_all=False): 
        super(CollarPlot, self).__init__(
            underlying_price,
            options=[long_put_option, short_call_option],
            strategy_label='Collar',
            plot_all=plot_all)


class LongComboPlot(OptionStrategyPlot):
    """
    The Long Combo is a variation of the Long Synthetic Future. The only difference is that we
    sell OTM (lower strike) puts and buy OTM (higher strike) calls. The net effect is an
    inexpensive trade, similar to a Long Stock or Long Futures position, except there is a gap
    between the strikes.

    Moneyness of the options –
        Sell 1 OTM Put (Lower Strike)
        Buy 1 OTM Call (Higher Strike)

    Maximum Profit: Unlimited

    Maximum Loss: Unlimited (Lower Strike + Net Premium)

    Breakeven: Higher Strike + Net Premium
    """
    def __init__(self, underlying_price, short_put_option, long_call_option, plot_all=False):
        super(LongComboPlot, self).__init__(
            underlying_price,
            options=[short_put_option, long_call_option],
            strategy_label='Long Combo',
            plot_all=plot_all)


class BullCallSpreadPlot(OptionStrategyPlot):
    """
    Bull call spreads "Long Call Vertical Spread" are an options strategy that involves
    purchasing call options at a specific strike price, while also writing the same number
    of calls on the same asset and expiration date but at a higher strike price. A bull call
    spread is used when a moderate rise in the price of the underlying asset is expected.

    Moneyness Of The Options:
        Buy 1 OTM Strike Call
        Sell 1 OTM Strike Call

    Maximum Profit: Strike Price of Short Call – Strike Price of Long Call-Net Premium Paid

    Maximum Loss: Net Premium Paid

    Breakeven: Strike Price of Long Call + Net Premium Paid
    """
    def __init__(self, underlying_price, long_call_option, short_call_option, plot_all=False):
        super(BullCallSpreadPlot, self).__init__(
            underlying_price,
            options=[long_call_option, short_call_option],
            strategy_label='Bull Call Spread',
            plot_all=plot_all)


class BearCallLadderPlot(OptionStrategyPlot):
    """
    The Bear Call Ladder is also known as the "Short Call Ladder" and is an extension to the
    Bear Call Spread. Although this is not a Bearish Strategy, it is implemented when one is
    bullish. It is usually set up for a ‘net credit’ and the cost of purchasing call options
    is financed by selling an ‘in the money’ call option.

    For this Options Trading Strategy, one must ensure the Call options belong to the same
    expiry, the same underlying asset and the ratio are maintained. It mainly protects the
    downside of a Call sold by insuring it i.e. by buying a Call of a higher strike price.
    It is essential though that you execute the strategy only when you are convinced that
    the market would be moving significantly higher.

    The Bear Call Ladder or Short Call Ladder is best to use when you are confident that an
    underlying security would be moving significantly. It is a limited risk and an unlimited
    reward strategy if the movement comes on the higher side.


    Moneyness Of The Options:
        Selling 1 ITM call option
        Buying 1 ATM call option
        Buying 1 OTM call option

    Executed in a 1:1:1 ratio combination, i.e. for every 1 ITM Call option sold, 1 ATM and
    1 OTM Call option has to be bought. (ie by selling 1 ITM, buying 1 ATM, and 1 OTM)

    Other possible combinations are 2:2:2 or 3:3:3 (so on and so forth).
    """
    def __init__(self, underlying_price, atm_long_call_option,
                 otm_long_call_option, short_call_option, plot_all=False):
        super(BearCallLadderPlot, self).__init__(
            underlying_price,
            options=[atm_long_call_option, otm_long_call_option, short_call_option],
            strategy_label='Bear Call Ladder', plot_all=plot_all)


class BearPutSpreadPlot(OptionStrategyPlot):
    """
    A bear put spread "Long Put Spread", also known as a bear put debit spread, is a type of
    options strategy used when an options trader expects a decline in the price of the
    underlying asset. A bear put spread is achieved by purchasing put options at a specific
    strike price while also selling the same number of puts with the same expiration date at
    a lower strike price. The maximum profit using this strategy is equal to the difference
    between the two strike prices, minus the net cost of the options.

    Moneyness Of The Options:
        Buy 1 ITM Put
        Sell 1 OTM Put
    """
    def __init__(self, underlying_price, long_put_option, short_put_option, plot_all=False):
        super(BearPutSpreadPlot, self).__init__(
            underlying_price,
            options=[long_put_option, short_put_option],
            strategy_label='Bear Put Spread',
            plot_all=plot_all)


class IronCondorPlot(OptionStrategyPlot):
    """
    The iron condor is an option trading strategy utilizing two vertical spreads – a put spread
    and a call spread with the same expiration and four different strikes. A long iron condor
    is essentially selling both sides of the underlying instrument by simultaneously shorting
    the same number of calls and puts, then covering each position with the purchase of further
    out of the money call(s) and put(s) respectively. The converse produces a short iron condor.

    Moneyness of the options:

        Sell 1 OTM Put (Higher Strike)
        Sell 1 OTM Call (Lower Strike)
        Buy 1 OTM Put (Lower Strike)
        Buy 1 OTM Call (Higher Strike)

    Maximum Profit: Net Premium Received

    Maximum Loss:

        Strike Price of Long Call – Strike Price of Short Call – Net Premium Received or
        Strike Price of Short Put – Strike Price of Long Put – Net Premium Received whichever is higher

    Breakeven:

        Upper side: Strike Price of Short Call + Net Premium Received
        Lower side: Strike Price of Short Put – Net Premium Received
    """
    def __init__(self, underlying_price, long_put_option, short_put_option,
                 long_call_option, short_call_option, plot_all=False):
        super(IronCondorPlot, self).__init__(
            underlying_price,
            options=[long_put_option, short_put_option, long_call_option, short_call_option],
            strategy_label='Iron Condor',
            plot_all=plot_all)


class LongCallButterflyPlot(OptionStrategyPlot):
    """
    A long butterfly spread with calls is a three-part strategy that is created by buying one
    call at a lower strike price, selling two calls with a higher strike price and buying
    one call with an even higher strike price.
    """
    def __init__(self, underlying_price, itm_long_call_option, atm_short_call_option,
                 otm_long_call_option, plot_all=False):
        super(LongCallButterflyPlot, self).__init__(
            underlying_price,
            options=[itm_long_call_option, atm_short_call_option, otm_long_call_option],
            strategy_label='Long Call Butterfly',
            plot_all=plot_all)


class CallCalendarSpreadPlot(OptionStrategyPlot):
    """
    Moneyness Of The Options:
        Sell 1 Front Month OTM Call (ex 03/21/2018)
        Buy 1 Back Month OTM Call (ex 04/21/2018)
        (Same strike price, about month apart, 1-2 steps away from the current price)

    """
    def __init__(self, underlying_price, otm_near_short_call_option,
                 otm_far_long_call_option, plot_all=False):
        super(CallCalendarSpreadPlot, self).__init__(
            underlying_price,
            options=[otm_near_short_call_option, otm_far_long_call_option],
            strategy_label='Calendar Spread',
            plot_all=plot_all)


class CreditPutSpreadPlot(OptionStrategyPlot):
    """
    Credit spreads involve the simultaneous purchase and sale of options contracts of
    the same class (puts or calls) on the same underlying security. In the case of a
    vertical credit put spread, the expiration month is the same, but the strike price
    will be different.

    Moneyness Of The Options:
        Buy 1 OTM Put (Lower Strike)
        Sell 1 ITM Put (Higher Strike)
    """
    def __init__(self, underlying_price, otm_long_put_option, itm_short_put_option, plot_all=False):
        super(CreditPutSpreadPlot, self).__init__(
            underlying_price,
            options=[otm_long_put_option, itm_short_put_option],
            strategy_label='Vertical Credit Put Spread',
            plot_all=plot_all)


class CreditCallSpreadPlot(OptionStrategyPlot):
    """
    Credit spreads involve the simultaneous purchase and sale of options contracts of
    the same class (puts or calls) on the same underlying security. In the case of a
    vertical credit put spread, the expiration month is the same, but the strike price
    will be different.

    Moneyness Of The Options:
        Sell 1 ITM Call (Lower Strike)
        Buy 1 OTM Call (Higher Strike)
    """
    def __init__(self, underlying_price, otm_long_call_option, itm_short_call_option, plot_all=False):
        super(CreditCallSpreadPlot, self).__init__(
            underlying_price,
            options=[otm_long_call_option, itm_short_call_option],
            strategy_label='Vertical Credit Call Spread',
            plot_all=plot_all)


class OptionsDataParser(object):

    @classmethod
    def parse_cboe_option_symbol(cls, name, is_call=True):
        start = 0
        value = name[name.find('(') + 1:name.find(')')]
        code = value
        for s in value:
            if s.isdigit():
                break
            start = start + 1
        symbol = value[:start]
        stop = start
        for s in value[start:]:
            if s.isalpha():
                break
            stop = stop + 1
        expiration = value[start:stop]
        # Mini option symbols end with 7
        # https://en.wikipedia.org/wiki/Option_symbol
        if len(expiration) > 4:
            mini_option = expiration[:1]
            expiration = expiration[1:]
        else:
            mini_option = None
        month = value[stop:stop + 1]
        value = value[stop + 1:]
        if '-' in value:
            start = value.find('-')
            price = value[:start]
            exchange = value[start + 1:]
        else:
            price = value
            exchange = None
        # Convert data to proper data types
        if is_call:
            month = call_month_codes[month]
            option_type = 'call'
        else:
            month = put_month_codes[month]
            option_type = 'put'

        return OptionCode(option_type=option_type,
                          symbol=symbol,
                          expiration=pd.to_datetime('{} {}, {} 23:59:59'.format(month, expiration[2:], expiration[:2])),
                          strike=float(price),
                          mini_option=True if mini_option != None else False,
                          exchange=cboe_exchanges[exchange] if exchange in cboe_exchanges else 'Unknown',
                          code=code,
                          default=name)

    @classmethod
    def parse_expiration_time(cls, item, plot_dates_dict=None, current_time=None, is_call=True):
        code = cls.parse_cboe_option_symbol(item, is_call=is_call)
        days_to_expiration = (code.expiration - current_time).days
        if plot_dates_dict != None and isinstance(plot_dates_dict, dict):
            value = code.expiration.strftime('%b %d %Y')
            if value not in plot_dates_dict:
                plot_dates_dict[value] = code.expiration
        return days_to_expiration

    @classmethod
    def calculate_expiration_time(cls, item):
        d1 = parser.parse(item['ExpirationDate'])
        d2 = parser.parse(item['DataDate'])
        days_to_expiration = (d1 - d2).days
        return days_to_expiration

    @classmethod
    def parse_strike_price(cls, item, is_call=True):
        code = cls.parse_cboe_option_symbol(item, is_call=is_call)
        return code.strike

    @classmethod
    def _apply_data_filter(cls, data, filter=None):
        if not filter:
            return data
        query = ' & '.join(['{} {} {}'.format(k, v[0], repr(v[1])) for k, v in filter.items()])
        return data.query(query)

    @classmethod
    def parse_options_data(cls, input_file, input_type=OptionDataInputTypes.cboe, symbols=None, data_date=None):
        if input_type == OptionDataInputTypes.cboe:
            # Parse 2 top lines from the file to read current stock info.
            info = []
            try:
                with open(input_file, 'r') as fd:
                    info.append(fd.readline())
                    info.append(fd.readline())
            except IOError:
                msg = "Unable to read data from {} options input file\n".format(input_file)
                logger.error(msg)
                raise RuntimeException(msg)
            except Exception as ex:
                msg = "An error occurred while opening {} options input file\n".format(input_file)
                logger.error(msg)
                raise RuntimeException(msg)

            logger.info("Parsing options info from file {}\n".format(input_file))
            logger.info("Underlying stock info: %s \n%s\n" % (info[0].strip(), info[1].strip()))

            value = info[0].split(',')
            name = value[0]
            symbol = name[:name.find(' ')]
            company = name[name.find('(') + 1:name.find(')')]
            underlying_price = float(value[1])

            value = info[1].split()
            current_time = pd.to_datetime('{} {} {} {}'.format(
                value[0], value[1], value[2], value[4]))
            # day_count = int(current_time.strftime('%j'))
            day_count = current_time.timetuple().tm_yday

            data = pd.io.parsers.read_csv(input_file, sep=',', header=2, na_values=' ')
            # Fix NA values in dataframe
            data = data.fillna(0.0)

            # Let's look at data where there was a recent sale
            # data = data[data.Calls > 0]

            # Column names cleanup
            data = data.rename(index=str, columns={
                'Last Sale': 'CallLastSale',
                'Last Sale.1': 'PutLastSale',
                'Ask': 'CallAsk',
                'Ask.1': 'PutAsk',
                'Bid': 'CallBid',
                'Bid.1': 'PutBid',
                'Net': 'CallNet',
                'Net.1': 'PutNet',
                'Vol': 'CallVolume',
                'Vol.1': 'PutVolume',
                'Open Int': 'CallOpenInterest',
                'Open Int.1': 'PutOpenInterest'
            })

            data = data[(data['CallLastSale'] > 0) | (data['PutLastSale'] > 0)]

            # Expiration date parser
            plot_dates_dict = {}
            pext = partial(cls.parse_expiration_time,
                           plot_dates_dict=plot_dates_dict,
                           current_time=current_time, is_call=True)
            exp = data.Calls.apply(pext)
            exp.name = 'Expiration'

            # Strike price parser
            psp = partial(cls.parse_strike_price, is_call=True)
            strike = data.Calls.apply(psp)
            strike.name = 'Strike'

            data = data.join(exp).join(strike)

            # Let's filter out expired options
            data = data[data['Expiration'] > 0]

            # TODO: Add support for multiple files
            options = []
            opt = OptionsData(data=data, plot_dates_dict=plot_dates_dict,
                              symbol=symbol, company=company,
                              underlying_price=underlying_price,
                              day_count=day_count, current_time=current_time)
            options.append(opt)
            return options
        elif input_type == OptionDataInputTypes.discountoptions:
            filter = {}
            if symbols:
                filter['Symbol'] = ['in', symbols]
            if data_date:
                filter['DataDate'] = ['==', data_date]

            dtype = {
                'Symbol': str,
                'StrikePrice': np.float64,
                'AskPrice': np.float64,
                'AskSize': np.float64,
                'BidPrice': np.float64,
                'BidSize': np.float64,
                'LastPrice': np.float64,
                'Volume': np.int32,
                'ImpliedVolatility': np.float64,
                'Delta': np.object,
                'Gamma': np.object,
                'Vega': np.object,
                'Rho': np.object,
                'OpenInterest': np.int32,
                'UnderlyingPrice': np.float64
            }

            size = os.path.getsize(input_file)
            # Check if we have > ~50MB file, and if true read file in chunks
            if size > 50000000:
                chunksize = 200000 # Lines
                data = pd.DataFrame()
                for chunk in pd.read_csv(input_file, chunksize=chunksize, dtype=dtype):
                    filtered = cls._apply_data_filter(chunk, filter=filter)
                    data = data.append(filtered)
            else:
                data = pd.io.parsers.read_csv(input_file, sep=',', header=0, na_values=' ', dtype=dtype)
                data = cls._apply_data_filter(data, filter=filter)

            # Fix NA values in dataframe and set multi index
            data = data.fillna(0.0)
            data = data.reset_index(drop=True).set_index(['Symbol', 'DataDate', 'ExpirationDate', 'StrikePrice'])

            # Break data in calls and puts and merge them back into single rows (it should mimic CBOE data format)
            calls = data[data['PutCall'] == 'call']
            calls = calls.rename(index=str, columns={
                'AskPrice': 'CallAsk',
                'AskSize': 'CallAskSize',
                'BidPrice': 'CallBid',
                'BidSize': 'CallBidSize',
                'LastPrice': 'CallLastSale',
                'Volume': 'CallVolume',
                'ImpliedVolatility': 'CallImpliedVolatility',
                'Delta': 'CallDelta',
                'Gamma': 'CallGamma',
                'Vega': 'CallVega',
                'Rho': 'CallRho',
                'OpenInterest': 'CallOpenInterest',
                'UnderlyingPrice': 'CallUnderlyingPrice' # Do we need this field? 
            })
            calls = calls.drop(['PutCall'], axis=1)

            puts = data[data['PutCall'] == 'put']
            puts = puts.rename(index=str, columns={
                'AskPrice': 'PutAsk',
                'AskSize': 'PutAskSize',
                'BidPrice': 'PutBid',
                'BidSize': 'PutBidSize',
                'LastPrice': 'PutLastSale',
                'Volume': 'PutVolume',
                'ImpliedVolatility': 'PutImpliedVolatility',
                'Delta': 'PutDelta',
                'Gamma': 'PutGamma',
                'Vega': 'PutVega',
                'Rho': 'PutRho',
                'OpenInterest': 'PutOpenInterest',
                'UnderlyingPrice': 'PutUnderlyingPrice' # Do we need this field? 
            })
            puts = puts.drop(['PutCall'], axis=1)
            data = calls.join(puts)
            data = data[(data['CallLastSale'] > 0) | (data['PutLastSale'] > 0)]

            # Rest index and add expiration column
            data = data.reset_index().rename(
                index=str,
                columns={'StrikePrice': 'Strike'})
            # Expiration date calculator
            exp = data.apply(cls.calculate_expiration_time, axis=1)
            exp.name = 'Expiration'
            data = data.join(exp)

            # Let's filter out expired options
            data = data[data['Expiration'] > 0]

            data['DataDate'] = pd.to_datetime(data['DataDate'])
            data['ExpirationDate'] = pd.to_datetime(data['ExpirationDate'])
            data['Strike'] = pd.to_numeric(data['Strike'])
            data = data.sort_values(by=['Symbol', 'DataDate'])

            # Index by symbol and data date
            data = data.reset_index(drop=True).set_index(['Symbol', 'DataDate'])

            options = []
            indices = set(data.index.values.tolist())

            for index in indices:
                current_time = pd.to_datetime(index[1])
                day_count = current_time.timetuple().tm_yday
                df = data.loc[index, :].reset_index().copy()

                # Calculate open interest
                put_open_interest = np.asscalar(df['PutOpenInterest'].sum())
                call_open_interest = np.asscalar(df['CallOpenInterest'].sum())

                # Call and put underlying price should be identical
                underlying_price = df[['CallUnderlyingPrice', 'PutUnderlyingPrice']].drop_duplicates().values[0][0]
                opt = OptionsData(data=df,
                                  plot_dates_dict={},
                                  symbol=index[0],
                                  company=None,
                                  underlying_price=underlying_price,
                                  day_count=day_count,
                                  current_time=current_time,
                                  call_open_interest=call_open_interest,
                                  put_open_interest=put_open_interest)
                options.append(opt)

            return options
        else:
            raise RuntimeException("Unsupported input format {}".format(input_type))

def normalize_path(path, default_extension='.png'):
    if not path:
        return path
    ext = os.path.splitext(path)[1]
    if not ext or ext == '.':
        path = path + (default_extension if ext != '.' else default_extension[1:])
    if os.path.isabs(path):
        return path
    return os.path.join(working_dir, path)


def exec_strategy_plot(args):
    # TODO: Implement strategy builder.
    # put_option = PutOption(65.0, 100, 2.0, 2.0, position=OptionPosition.long)
    # call_option = CallOption(75.0, 100, 3.25, 3.25, position=OptionPosition.short)
    # collar = CollarPlot(70.65, put_option, call_option)
    # collar.plot()
    pass


def exec_strategy_selector(args):
    term = implied_volatility_terms[args.iv_term]
    options = OptionsDataParser.parse_options_data(args.input_file, input_type=args.input_type,
                                                   symbols=[args.symbol], data_date=args.data_date)
    for chain in options:
        for flag in ['c', 'p',]:
            iv = ImpliedVolatilityHelper.weighted_iv(chain, term, flag)
            print(chain.symbol, args.iv_term, flag, chain.current_time, iv)


def exec_iv_calculate_historic(args):
    manager = Manager(db_uri=args.database_uri)

    if args.update_all_symbols:
        symbols = [symbol[0] for symbol in manager.get_iv_symbols()]
    else:
        symbols = args.symbols

    print('Updating {} for following symbols {}...'.format(args.iv_term, ', '.join(symbols)))

    files = []
    for file in args.dir_or_file:
        if os.path.isfile(file):
            files.append(file)
        elif os.path.isdir(file):
            path = os.path.join(file, "*.csv")
            files = files + glob.glob(path)
    term = implied_volatility_terms[args.iv_term]

    print('Processing {} files...'.format(len(files)))
    for file in sorted(files):
        print('Processing input file {}...'.format(file))
        options = OptionsDataParser.parse_options_data(file, input_type=args.input_type,
                                                       symbols=symbols,
                                                       data_date=args.data_date)
        for chain in options:
            # Convert pandas timestamp to datetime
            day = chain.current_time.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
            data = {
                'day': day,
                'symbol': chain.symbol,
                'term': args.iv_term,
                'put_open_interest': chain.put_open_interest,
                'call_open_interest': chain.call_open_interest
            }
            for option_type in ['call', 'put',]:
                flag = option_type.lower()[0]
                iv = ImpliedVolatilityHelper.weighted_iv(chain, term, flag)
                data[option_type] = iv
            entity = manager.create_iv(data)


def exec_iv_files_plot(args):
    options = OptionsDataParser.parse_options_data(args.input_file, input_type=args.input_type)
    for chain in options:
        ImpliedVolatilityHelper.plot(chain, args.matplotlib_style,
                                     historic_volatility=args.historic_volatility,
                                     output_image=normalize_path(args.output_image))


def exec_iv_db_list(args):
    manager = Manager(db_uri=args.database_uri)
    for symbol in manager.get_iv_symbols():
        print(symbol[0])


def exec_iv_db_plot(args):
    manager = Manager(db_uri=args.database_uri)
    iex_columns=[
        'date',
        'open', 'high',
        'low', 'close',
        'volume', 'unadjustedVolume',
        'change', 'changePercent',
        'vwap', 'label',
        'changeOverTime']
    stocks = pd.DataFrame(columns=iex_columns)
    if args.historic_volatility:
        charts = StockReader(symbols=args.symbols).get_chart(range='2y')
        if charts:
            if isinstance(charts, dict):
                for key, value in charts.items():
                    df = pd.DataFrame(value, columns=iex_columns)
                    df['symbol'] = key
                    df = ImpliedVolatilityHelper.calculate_returns(df, close='close')
                    stocks = stocks.append(df, ignore_index=True)
            else:
                stocks = stocks.append(charts, ignore_index=True)
                stocks['symbol'] = args.symbols[0]
                stocks = ImpliedVolatilityHelper.calculate_returns(stocks, close='close')

            stocks['date'] = pd.to_datetime(stocks['date'])
            stocks = stocks.reset_index(drop=True).set_index(['date'])

    df = pd.DataFrame(columns=['day', 'symbol', 'call', 'put', 'term'])
    for symbol in args.symbols:
        entities = manager.get_iv_by_symbol_and_term(symbol, args.iv_term)
        for entity in entities:
            df = df.append(entity.serialize_data_frame, ignore_index=True)

    df = df.reset_index(drop=True).set_index(['day'])

    fig, ax = plt.subplots(figsize=(14, 8))
    color = 'tab:blue'
    ax.set_ylabel('Volatility', color=color)
    ax.set_title(args.iv_term.upper() + ' Index Mean')

    ax.yaxis.set_major_locator(plticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(0.05))

    ax.yaxis.grid(True, 'minor', linewidth=0.5, linestyle='-.')
    ax.yaxis.grid(True, 'major', linewidth=0.5)

    ax.tick_params(labelcolor=color, labeltop=False, labelright=False)
    ax.set_xlabel('Date')

    for key, data in df.groupby(['symbol']):
        group = data.copy()
        group[args.iv_term] = group[['call', 'put']].mean(axis=1)
        group[args.iv_term].plot(ax=ax, grid=True, color=color, kind='line', alpha=0.8, label=key + ' IV (' + args.iv_term.upper() + ')')
        if stocks.size > 0:
            # Plot HV chart
            color = 'tab:orange'
            # We are running inner join to eliminate weekends from the options data
            group = pd.concat([group, stocks[(stocks['symbol'] == key)][['Volatility', 'close']]], axis=1, join='inner')
            # group = group.join(stocks[(stocks['symbol'] == key)][['Volatility', 'close']])
            group['Volatility'].plot(ax=ax, grid=True,
                                     kind='line', color=color, linestyle='--',
                                     linewidth=1, alpha=0.7, label=key + ' HV')
            # Plot stock chart
            color = 'tab:green'
            ax2 = ax.twinx()  # Create second axes that shares the same x-axis
            ax2.set_ylabel('Stock Price', color=color)  # The x-label is handled with ax
            group['close'].plot(ax=ax2, grid=False, kind='line', color=color, linewidth=2, alpha=0.5, label=key + ' Stock')
            ax2.tick_params(axis='y', labelcolor=color, labelright=True)
            ax2.margins(0.01)
            ax2.legend(loc='best')

    ax.margins(0.01)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()
