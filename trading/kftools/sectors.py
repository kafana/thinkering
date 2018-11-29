"""
https://iextrading.com/
https://iextrading.com/developer/docs/
https://addisonlynch.github.io/iexfinance/stable/index.html
"""

import os
import sys
import math
import argparse
import enum
import logging
from collections import namedtuple
from functools import partial
from datetime import datetime, timedelta


import iexfinance
from iexfinance import Stock
from iexfinance import get_available_symbols
from iexfinance import (get_market_tops, get_market_gainers,
                        get_market_losers, get_market_most_active,
                        get_market_iex_volume, get_market_iex_percent)
from iexfinance.base import _IEXBase
from iexfinance.stock import HistoricalReader #, MoversReader


from kftools.models import Manager, DatabaseClient, Industry, Sector, Tag, Company


PARSER_COMMANDS_MODEL_MAP = {
    'company': Company,
    'sector': Sector,
    'tag': Tag,
    'industry': Industry}


class MoversReader(_IEXBase):
    """
    Base class for retrieving market movers from the Stocks List endpoint

    Parameters
    ----------
    mover: str
        Desired mover
    """
    _AVAILABLE_MOVERS = ["mostactive", "gainers", "losers", "iexvolume",
                         "iexpercent", "infocus"]

    def __init__(self, mover=None, **kwargs):
        super(MoversReader, self).__init__(**kwargs)
        if mover in self._AVAILABLE_MOVERS:
            self.mover = mover
        else:
            raise ValueError("Please input a valid market mover.")

    @property
    def url(self):
        return 'stock/market/list/' + self.mover


def get_market_in_focus(**kwargs):
    """
    Top-level function for obtaining the 10 symbols with the highest
    percent change on the IEX exchange from the Stocks list endpoint
    """
    return MoversReader(mover='infocus', **kwargs).fetch()


class MarketSectorPerformanceReader(_IEXBase):

    def __init__(self, **kwargs):
        super(MarketSectorPerformanceReader, self).__init__(**kwargs)

    @property
    def url(self):
        return 'stock/market/sector-performance'


def exec_database_actions(args):
    if args.rebuild_database and args.database_uri.startswith('sqlite://'):
        file = args.database_uri[len('sqlite://'):]
        file = file[1:] if file[:2] == '//' else file
        if os.path.isfile(file):
            if not DatabaseClient.is_sqlite_db(file):
                raise RuntimeError('Unable to continue with rebuilding database. '
                                   'The {} file seems to be invalid sqlite db'.format(file))
            os.remove(file)

    manager = Manager(db_uri=args.database_uri)

    # Populate initial sectors
    sectors = {sector['name']:[] for sector in MarketSectorPerformanceReader().fetch()}
    processed = total = 0
    sys.stdout.write('Loading data into {} database\n'.format(args.database_uri))
    for symbol in get_available_symbols():
        # Only parse cs and et types (common stock and ETFs)
        if symbol['type'] == 'cs' or symbol['type'] == 'et':
            company = Stock(symbol['symbol'], output_format='json').get_company()
            manager.create_iex_company(company)

            processed += 1
            if processed % 25 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        total += 1
        if total % 1000 == 0:
            sys.stdout.write('[{} of {}]\n'.format(processed, total))

        if total == 2500:
            break

    if total % 1000 != 0:
        sys.stdout.write('[{} of {}]\n'.format(processed, total))

    sys.stdout.write('{} of {} records loaded to {} database\n'.format(
        processed, total, args.database_uri))
    sys.stdout.flush()


def _build_filters(args):
    filters = {}
    if not args.all:
        if args.name:
            filters['name'] = args.name
        if args._id:
            filters['id'] = int(args._id)
        if 'symbol' in args and args.symbol:
            filters['symbol'] = args.symbol
    return filters


def exec_report_actions(args):
    manager = Manager(db_uri=args.database_uri)

    if args.command == 'company':
        if args.action == 'list':
            for company in manager.fetch_collection_for_class(
                    Company,
                    order_by=args.order_by,
                    is_asc=True,
                    limit=args.limit,
                    **_build_filters(args)):
                print(company.serialize_terminal)
    else:
        if args.action == 'list':
            if args.list_companies:
                if args.command == 'industry':
                    for company in manager.get_companies_by_industry(
                            order_by=args.order_by, is_asc=True,
                            limit=args.limit, **_build_filters(args)):
                        print(company.serialize_terminal)
                elif args.command == 'tag':
                    for company in manager.get_companies_by_tag(
                            order_by=args.order_by, is_asc=True,
                            limit=args.limit, **_build_filters(args)):
                        print(company.serialize_terminal)
                elif args.command == 'sector':
                    for company in manager.get_companies_by_sector(
                            order_by=args.order_by, is_asc=True,
                            limit=args.limit, **_build_filters(args)):
                        print(company.serialize_terminal)
            else:
                for item in manager.fetch_collection_for_class(
                        PARSER_COMMANDS_MODEL_MAP[args.command],
                        order_by=args.order_by,
                        is_asc=True,
                        limit=args.limit,
                        **_build_filters(args)):
                    print(item.serialize_terminal)


def exec_market_actions(args):
    manager = Manager(db_uri=args.database_uri)

    if args.movers:
        fn = getattr(iexfinance, 'get_market_{}'.format(args.movers))
        movers = fn()
        if movers:
            movers.sort(key=lambda e: e['changePercent'], reverse=True)
            print('symbol\t\topen\t\tlatest\t\tprevious\tchange\t\tpercent\t\tname')
            for mover in movers:
                print('{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{:.2f}%\t\t{}'.format(
                    mover['symbol'], mover['open'], mover['latestPrice'],
                    mover['previousClose'], mover['change'],
                    mover['changePercent'] * 100, mover['companyName']))


def exec_stock_actions(args):
    manager = Manager(db_uri=args.database_uri)

    if args.peers:
        peers = Stock(args.symbol).get_peers()
        for symbol in peers:
            company = manager.get_company_by_symbol(symbol)
            print(company.serialize_terminal)
