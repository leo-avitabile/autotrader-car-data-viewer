import datetime
import json
import os
import pathlib
from typing import Dict, Union

import pandas as pd
import logging
from hashlib import md5
import sqlite3

import glob
from json.decoder import JSONDecodeError

"""
Store objects keyed by the hash.
"""

LOGGER = logging.getLogger(__name__)


def string_md5(string_to_hash: str) -> str:
    return md5(string_to_hash.encode('ascii')).hexdigest()


class DatabaseManager:

    def __init__(self, path='autotrader_cache5.json'):
        # open and load the data if it exists
        self.path = path
        self.current_data = {}
        self.current_keys = set()
        if os.path.isfile(self.path):
            with open(self.path) as f:
                # may fail to open
                try:
                    self.current_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    pass  # if failure is detected, use uninitalised data

        LOGGER.info(f'{self.__class__.__name__} initialised with {len(self.current_data)} objects')

    def append_snapshot(self, df: pd.DataFrame) -> None:
        start = datetime.datetime.now()
        # generate hashes for all of the rows in the df
        df['hash'] = df['url'].apply(string_md5)
        LOGGER.debug(f'append_snapshot called on {len(df)} objects')

        # get the car make/model, convert to lowercase to ensure that weird user input is always filed away safely
        make = set(df['make'].to_list()).pop().lower()
        model = set(df['model'].to_list()).pop().lower()

        # create slots for makes and models that don't exist yet
        if make not in self.current_data:
            self.current_data[make] = {}
        if model not in self.current_data[make]:
            self.current_data[make][model] = list()

        # generate a mapping of hashes to the list entries that contain them
        hashes = {x['hash']: x for x in self.current_data[make][model]}  # gets a reference to the dict in the list

        # generate a single timestamp to save doing it every time below, it will be used below
        now = datetime.datetime.now().utcnow().timestamp()

        for _, row in df.iterrows():

            hash_val = row['hash']
            name = row['name']

            if hash_val in hashes:
                LOGGER.debug(f'append_snapshot updating "last_seen" attrib of {name} to {now}')
                hashes[hash_val]['last_seen'] = now
                for k, v in json.loads(row.to_json()).items():
                    if k not in hashes[hash_val]:
                        # assert type(v) in (int, str)  # Todo: Saw a NoneType here once. Maybe fix that
                        hashes[hash_val][k] = v.lower() if type(v) is str else v
            else:
                LOGGER.debug(f'append_snapshot adding entry: {name}')
                data = json.loads(row.to_json())
                data['first_seen'] = now
                self.current_data[make][model].append(data)
                continue

        after_update = datetime.datetime.now()
        LOGGER.debug('Update took', (after_update - start).total_seconds(), 's')

        with open(self.path, 'w') as f:
            json.dump(self.current_data, f, indent=2)

        after_save = datetime.datetime.now()
        LOGGER.debug('Save took', (after_save - after_update).total_seconds(), 's')

    def fetch(self, make, model, **extras):
        make_lower = make.lower()
        model_lower = model.lower()
        if make_lower in self.current_data and model_lower in self.current_data[make_lower]:

            # if no extras are specified, return the lot
            if not extras:
                return self.current_data[make_lower][model_lower]

            # if extras are specified, check and add
            results = []
            for row in self.current_data[make_lower][model_lower]:
                for k, v in extras.items():

                    # try and get the param from the database
                    val = row.get(k.lower(), None)
                    if val is None:
                        # if it does not exist (i.e. none is returned, break)
                        break

                    # check the value against the filter request
                    # if they are different then break
                    val_normalised = val.lower() if type(val) is str else val
                    if val_normalised != v.lower():
                        break
                else:
                    # no breaks hit
                    # if all(row.get(k.lower(), None).lower() == v.lower() for k, v in extras.items()):
                    results.append(row)
            return results
        return None


class DatabaseManager2:

    # B_FILE_NAME = 'cars.sqlite'
    TABLE_NAME = 'tbl_cars'

    def __init__(self, db_name='test.db'):
        self.conn = None
        self.cursor = None
        pass

    def append_snapshot(self, df: pd.DataFrame) -> None:
        self.conn = sqlite3.connect(self.DB_FILE_NAME)
        self.cursor = self.conn.cursor()
        df['hash'] = df['url'].apply(string_md5)
        LOGGER.debug(f'append_snapshot called on {len(df)} objects')
        df.to_sql('df', self.conn, if_exists='append')
        self.conn.commit()

    def fetch(self, make, model, **extras):
        pass


class DatabaseManager3:
    '''Same approach as the initial config but uses separate files to speed up save time'''

    def __init__(self, directory='cache'):
        # open and load the data if it exists
        os.makedirs(directory, exist_ok=True)  # create the directory if it does not already exist
        self.directory = pathlib.Path(directory)

        # glob for files, they will be named <str(brand).lower()>.json
        files = glob.glob(str(self.directory / '*.json'))
        self.file_paths_by_brand = {}
        for file in files:
            file_name = os.path.basename(file)
            brand, _ = file_name.rsplit('.', maxsplit=1)
            self.file_paths_by_brand[brand] = os.path.abspath(file)

        self.directory = pathlib.Path(directory)

        LOGGER.info(f'{self.__class__.__name__} initialised with {len(files)} files')

    def append_snapshot(self, df: pd.DataFrame) -> None:
        start = datetime.datetime.now()
        # generate hashes for all of the rows in the df
        df['hash'] = df['url'].apply(string_md5)
        LOGGER.debug(f'append_snapshot called on {len(df)} objects')

        # get the car make/model, convert to lowercase to ensure that weird user input is always filed away safely
        make = set(df['make'].to_list()).pop().lower()
        model = set(df['model'].to_list()).pop().lower()

        # create slots for makes and models that don't exist yet
        if make not in self.file_paths_by_brand:
            file_path = os.path.abspath(self.directory / f'{make}.json')
            with open(file_path, 'w') as f:
                json.dump({}, f)
            self.file_paths_by_brand[make] = file_path
            current_data = {}
        else:
            try:
                # load the brands file
                with open(self.file_paths_by_brand[make]) as f2:
                    current_data = json.load(f2)
            except JSONDecodeError as e:
                current_data = {}

        # check the brand file has the data for the current model
        if model not in current_data:
            current_data[model] = list()

        # generate a mapping of hashes to the list entries that contain them
        hashes = {x['hash']: x for x in current_data[model]}  # gets a reference to the dict in the list

        # generate a single timestamp to save doing it every time below, it will be used below
        now = datetime.datetime.now().utcnow().timestamp()

        for _, row in df.iterrows():

            hash_val: str = row['hash']
            title: str = row['title']

            if hash_val in hashes:
                LOGGER.debug(f'append_snapshot updating "last_seen" attrib of {title} to {now}')
                hashes[hash_val]['last_seen'] = now
                for k, v in json.loads(row.to_json()).items():
                    if k not in hashes[hash_val]:
                        # assert type(v) in (int, str)  # Todo: Saw a NoneType here once. Maybe fix that
                        hashes[hash_val][k] = v.lower() if type(v) is str else v
            else:
                LOGGER.debug(f'append_snapshot adding entry: {title}')
                data = json.loads(row.to_json())
                data['first_seen'] = now
                current_data[model].append(data)

        after_update = datetime.datetime.now()
        LOGGER.debug('Update took %ss', (after_update - start).total_seconds())

        with open(self.file_paths_by_brand[make], 'w') as f:
            json.dump(current_data, f, indent=2)

        after_save = datetime.datetime.now()
        LOGGER.debug('Save took %ss', (after_save - start).total_seconds())

    def fetch(self, make: str, model: str, **extras: Dict[str, Union[str, int]]):
        make_lower = make.lower()
        model_lower = model.lower()
        if make_lower in self.file_paths_by_brand:

            data = None
            with open(self.file_paths_by_brand[make_lower]) as f:
                data = json.load(f)

            if model_lower not in data:
                return []

            # if no extras are specified, return the lot
            if not extras:
                return data[model_lower]

            # if extras are specified, check and add
            results = []
            for row in data[model_lower]:
                for k, v in extras.items():

                    # try and get the param from the database
                    val = row.get(k.lower(), None)
                    if val is None:
                        # if it does not exist (i.e. none is returned, break)
                        break

                    # check the value against the filter request
                    # if they are different then break
                    val_normalised = val.lower() if type(val) is str else val
                    if val_normalised != v.lower():
                        break
                else:
                    # no breaks hit
                    # if all(row.get(k.lower(), None).lower() == v.lower() for k, v in extras.items()):
                    results.append(row)
            return results
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    manager = DatabaseManager3()
    df = pd.DataFrame.from_dict([{'url': '1234'}])
    manager.append_snapshot(df)