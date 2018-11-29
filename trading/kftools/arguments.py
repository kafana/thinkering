import os
import argparse


import matplotlib.pyplot as plt


from kftools.sectors import (exec_database_actions,
                             exec_market_actions,
                             exec_stock_actions,
                             exec_report_actions)
from kftools.sectors import PARSER_COMMANDS_MODEL_MAP
from kftools.options import (exec_strategy_plot,
                             exec_strategy_selector,
                             exec_iv_calculate_historic,
                             exec_iv_files_plot,
                             exec_iv_db_plot,
                             exec_iv_db_list)
from kftools.options import OptionDataInputTypes


working_dir = os.getcwd()
working_home_dir = os.path.basename(working_dir)


DEFAULT_SQLALCHEMY_DATABASE_NAME = 'trading-tools.db'
DEFAULT_SQLALCHEMY_DATABASE_URI = 'sqlite:///{}/{}'.format(
    working_dir, DEFAULT_SQLALCHEMY_DATABASE_NAME)

ORDER_BY_FIELDS = ['name', 'id', 'created', 'modified']

MARKET_MOVERS_COMMANDS = ['most_active', 'gainers', 'losers',
                          'iex_volume', 'iex_percent', 'infocus']

ENTITY_COMMANDS = ['list', 'update', 'delete']
RECORD_LIMIT = 100

IMPLIED_VOLATILITY_TERMS_DEFULT = ['iv30', 'iv60', 'iv90', 'iv120']

MATPLOTLIB_STYLES_DEFULT = ['default', 'seaborn', 'classic']
MATPLOTLIB_STYLES = MATPLOTLIB_STYLES_DEFULT + sorted(
    style for style in plt.style.available if style not in MATPLOTLIB_STYLES_DEFULT)


def create_sectors_parser(parser):
    sectors = parser.add_parser('sectors', help='IEX helper tools')
    sectors.add_argument('-u', '--database-uri', default=DEFAULT_SQLALCHEMY_DATABASE_URI,
                         help='Database URI, defaults to `{}`'.format(
                             DEFAULT_SQLALCHEMY_DATABASE_URI))

    subparsers = sectors.add_subparsers(title='commands',
                                        dest='command',
                                        description='Sector utility. Needs an access to '
                                                    'IEX API `https://iextrading.com/'
                                                    'developer/docs/`.')

    # Database parser
    db_parser = subparsers.add_parser('db')
    db_parser.add_argument('-r', '--rebuild-database', action='store_true',
                           help='Destroys and rebuilds database. '
                                'It takes long time to complete this action.')
    db_parser.set_defaults(func=exec_database_actions)

    market_parser = subparsers.add_parser('market')
    market_parser.add_argument('-s', '--symbol', type=str,
                               help='Stock symbol')
    market_parser.add_argument('-m', '--movers', choices=MARKET_MOVERS_COMMANDS,
                               help='Market movers')
    market_parser.set_defaults(func=exec_market_actions)

    stock_parser = subparsers.add_parser('stock')
    stock_parser.add_argument('-s', '--symbol', type=str,
                              help='Stock symbol')
    group = stock_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--peers', action='store_true',
                       help='Shows peer tickers as defined by IEX')
    stock_parser.set_defaults(func=exec_stock_actions)

    # Commands parsers (companies, tags, sectors, industries)
    for command in PARSER_COMMANDS_MODEL_MAP.keys():
        command_parser = subparsers.add_parser(command)

        group = command_parser.add_mutually_exclusive_group(required=True)
        group.add_argument('-n', '--name', type=str, help='Name of the {}'.format(command))
        group.add_argument('-i', '--id', type=str, help='ID of the {}'.format(command), dest='_id')
        group.add_argument('-a', '--all', action='store_true', help='List all')
        if command == 'company':
            group.add_argument('-s', '--symbol', type=str, help='Search by company symbol')

        command_parser.add_argument('--action', choices=ENTITY_COMMANDS,
                                    default=ENTITY_COMMANDS[0],
                                    help='Action to run, defaults to `{}`'.format(
                                        ENTITY_COMMANDS[0]))
        command_parser.add_argument('--limit', type=int, default=RECORD_LIMIT,
                                    help=('Maximum number of records '
                                          'to return, defaults to `{}`').format(RECORD_LIMIT))
        command_parser.add_argument('--order-by', type=str, choices=ORDER_BY_FIELDS,
                                    default=ORDER_BY_FIELDS[0],
                                    help='Sort by field, defaults to `{}`'.format(
                                        ORDER_BY_FIELDS[0]))
        command_parser.set_defaults(func=exec_report_actions)

        # Don't add reverse companies search for company parser
        if command != 'company':
            command_parser.add_argument('-c', '--list-companies', action='store_true',
                                        help='List companies for {}'.format(command))

    return sectors


def create_strategy_parser(parser):
    strategy = parser.add_parser('strategy', help='Trading strategy helper tools')
    strategy.add_argument('-u', '--database-uri', default=DEFAULT_SQLALCHEMY_DATABASE_URI,
                          help='Database URI, defaults to `{}`'.format(
                              DEFAULT_SQLALCHEMY_DATABASE_URI))

    subparsers = strategy.add_subparsers(title='commands',
                                         dest='command',
                                         description='Strategy commands')

    sp = subparsers.add_parser('selector', help='Strategy selector')
    sp.add_argument('-t', '--input-type', type=OptionDataInputTypes,
                    choices=list(OptionDataInputTypes),
                    default=OptionDataInputTypes.cboe,
                    help='Input file type. Currently it '
                         'supports CBOE or DiscountOptions data only.')
    sp.add_argument('-T', '--iv-term', choices=IMPLIED_VOLATILITY_TERMS_DEFULT,
                    default=IMPLIED_VOLATILITY_TERMS_DEFULT[0],
                    help='Implied volatility terms')
    sp.add_argument('input_file', nargs='?', default='input.dat')
    sp.add_argument('-s', '--symbol', required=False,
                    help='Underlying symbol CSV file filter')
    sp.add_argument('-d', '--data-date', required=False,
                    help='Data date CSV file filter')
    sp.set_defaults(func=exec_strategy_selector)

    sp = subparsers.add_parser('plot', help='Strategy plot')
    sp.add_argument('-n', '--strategy-name', required=False, default='My options strategy',
                    help='Name of the options strategy')
    sp.add_argument('-u', '--underlying-price', type=float, required=True,
                    help='Price of the underlying stock')
    sp.set_defaults(func=exec_strategy_plot)

    return strategy


def create_iv_parser(parser):
    iv = parser.add_parser('iv', help='Implied volatility and options tools')
    iv.add_argument('-u', '--database-uri', default=DEFAULT_SQLALCHEMY_DATABASE_URI,
                    help='Database URI, defaults to `{}`'.format(
                        DEFAULT_SQLALCHEMY_DATABASE_URI))

    subparsers = iv.add_subparsers(title='commands',
                                   dest='command',
                                   description='IV commands')

    sp = subparsers.add_parser('build-db', help='Populate IV table')
    sp.add_argument('-t', '--input-type', type=OptionDataInputTypes,
                    choices=list(OptionDataInputTypes),
                    default=OptionDataInputTypes.cboe,
                    help='Input file type. Currently it supports '
                         'CBOE or DiscountOptions data only.')
    sp.add_argument('-T', '--iv-term', choices=IMPLIED_VOLATILITY_TERMS_DEFULT,
                    default=IMPLIED_VOLATILITY_TERMS_DEFULT[0], help='Implied volatility terms')
    group = sp.add_mutually_exclusive_group(required=True)
    group.add_argument('-A', '--update-all-symbols', action='store_true',
                       help='Updates all symbols currently present in IV table')
    group.add_argument('-s', '--symbols', nargs='+', help='Underlying symbols')
    sp.add_argument('-d', '--data-date', required=False,
                    help='Data date used for CSV file filter')
    sp.add_argument('-f', '--dir-or-file', nargs='+', default='data',
                    help='Input file or directory')
    sp.set_defaults(func=exec_iv_calculate_historic)

    sp = subparsers.add_parser('plot-db', help='Plot data from database')
    sp.add_argument('-s', '--symbols', required=True, nargs='+', help='Underlying symbols')
    sp.add_argument('-T', '--iv-term', choices=IMPLIED_VOLATILITY_TERMS_DEFULT,
                    default=IMPLIED_VOLATILITY_TERMS_DEFULT[0], help='Implied volatility terms')
    sp.add_argument('-H', '--historic-volatility', required=False, action='store_true',
                    help='Include historic volatility graph')
    sp.set_defaults(func=exec_iv_db_plot)

    sp = subparsers.add_parser('plot-files', help='Create IV graphs based on options input data')
    sp.add_argument('-t', '--input-type', type=OptionDataInputTypes,
                    choices=list(OptionDataInputTypes),
                    default=OptionDataInputTypes.cboe,
                    help='Input file type. Currently it '
                         'supports CBOE or DiscountOptions data only.')
    sp.add_argument('-H', '--historic-volatility', help='Include historic volatility graph',
                    action='store_true')
    sp.add_argument('-o', '--output-image', required=False, help='Name of the output image file')
    sp.add_argument('-s', '--matplotlib-style',
                    choices=MATPLOTLIB_STYLES,
                    default=MATPLOTLIB_STYLES[0],
                    help='Matplotlib style')
    sp.add_argument('input_file', nargs='?', default='input.dat')
    sp.set_defaults(func=exec_iv_files_plot)

    sp = subparsers.add_parser('list-db', help='List IV table records')
    sp.add_argument('-s', '--list-symbols', help='List IV table symbols', action='store_true')
    sp.set_defaults(func=exec_iv_db_list)

    return sp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(prog='kftools',
                                     description='Options and misc trading utilities')

    subparsers = parser.add_subparsers(title='commands', dest='command', help='Misc trading tools')

    sp = create_sectors_parser(subparsers) # Sectors parser
    sp = create_iv_parser(subparsers) # IV parser
    sp = create_strategy_parser(subparsers) # Strategy parser

    return parser.parse_args(args=args)
