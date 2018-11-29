import os
import sys
import argparse
from datetime import datetime

import pandas as pd

from kftools.zipline.ingest import load_from

working_dir = os.getcwd()
working_home_dir = os.path.basename(working_dir)
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
zipline_csv_dir_name = 'iex-csv-bundle'

EXPORT_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'dividend', 'split']

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(prog='zipline-csv-bundle-exporter',
                                     description='Exports IEX data to zipline CSV bundle format')
    parser.add_argument('-o', '--output-directory', help='Zipline CSV output directory',
                        default=working_dir + '/'+ zipline_csv_dir_name)
    parser.add_argument('-s', '--symbols', nargs='+', type=str, help='Stock symbols to export')
    parser.add_argument('-f', '--from-date', type=str, help='Start date in %%Y-%%m-%%d format')
    parser.add_argument('-t', '--to-date', type=str, help='End date in %%Y-%%m-%%d format')

    return parser.parse_args(args=args)

def export_data(panel, output_directory, timeframe='daily'):
    """
    Timeframe can be 'daily' or 'minute'.

    Bundle directory structure:
        <directory>/<timeframe1>/<symbol1>.csv
        <directory>/<timeframe1>/<symbol2>.csv
        <directory>/<timeframe1>/<symbol3>.csv
        <directory>/<timeframe2>/<symbol1>.csv
        <directory>/<timeframe2>/<symbol2>.csv
        <directory>/<timeframe2>/<symbol3>.csv
    """
    if not os.path.exists(output_directory):
        for label, data in panel.iteritems():
            start, end = pd.to_datetime(data.index.min()).strftime('%Y-%m-%d'), pd.to_datetime(data.index.max()).strftime('%Y-%m-%d')
            csvdir = os.path.join(output_directory, timeframe)
            os.makedirs(csvdir, exist_ok=True)
            csvfile = os.path.join(csvdir, '{}.csv'.format(label))
            print('Exporting {} bundle file [start={}, end={}]...'.format(csvfile, start, end))
            if 'dividend' not in data.columns:
                data['dividend'] = 0.0
            if 'split' not in data.columns:
                data['split'] = 1.0
            data.to_csv(path_or_buf=csvfile)
    else:
        raise RuntimeError('Directory {} already exists. Please '
                           'change output directory'.format(output_directory))


def main():
    args = parse_arguments()

    start = pd.Timestamp(args.from_date, tz='utc') if args.from_date else None
    end = pd.Timestamp(args.to_date, tz='utc') if args.to_date else None

    panel = load_from(args.symbols, start=start, end=end)
    export_data(panel, args.output_directory)


if __name__ == '__main__':
    main()
