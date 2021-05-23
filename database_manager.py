import datetime
import json
import os
import pandas as pd
import logging
from hashlib import md5

"""
Store objects keyed by the hash.
"""


def string_md5(string_to_hash: str) -> str:
    return md5(string_to_hash.encode('ascii')).hexdigest()


class DatabaseManager:

    def __init__(self, path='autotrader_cache4.json'):
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

        logging.info(f'{self.__class__.__name__} initialised with {len(self.current_data)} objects')

    def append_snapshot(self, df: pd.DataFrame) -> None:
        # generate hashes for all of the rows in the df
        df['hash'] = df['link'].apply(string_md5)
        logging.debug(f'append_snapshot called on {len(df)} objects')

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
                logging.debug(f'append_snapshot updating "last_seen" attrib of {name} to {now}')
                hashes[hash_val]['last_seen'] = now
                for k, v in json.loads(row.to_json()).items():
                    if k not in hashes[hash_val]:
                        assert type(v) in (int, str)
                        hashes[hash_val][k] = v.lower() if type(v) is str else v
            else:
                logging.debug(f'append_snapshot adding entry: {name}')
                data = json.loads(row.to_json())
                data['first_seen'] = now
                self.current_data[make][model].append(data)
                continue

        with open(self.path, 'w') as f:
            json.dump(self.current_data, f, indent=2)

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    m = DatabaseManager('autotrader_cache2.json')