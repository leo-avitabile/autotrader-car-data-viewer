# Todo:
# Fix charts randomly failing to draw - might be related to threading issues, has been better since removing printing
# Fetch from db in worker
# Add other scrape options - some done
# Add "new car" markers to graph - need to rethink the saving strategy to be able to do this
# Add start search on enter key


import sys
import logging
import re
import scraper as autotrader_scraper  # a hacked version of https://pypi.org/project/autotrader-scraper/
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from PySide2 import QtCore, QtWidgets, QtGui  # Downgraded to PySide2 (Qt5) for proper matplotlib compatibility
from PySide2.QtCore import QAbstractTableModel
from functools import lru_cache
from datetime import datetime
import database_manager
import webbrowser
import pathlib
from functools import partial
import matplotlib._color_data as mcd

supports_mapping = True

# try to import mapping libs, but do not require them
try:
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
except ImportError:
    cartopy_url = 'https://scitools.org.uk/cartopy/docs/latest/installing.html'
    logging.warning(f'To use mapping features please install cartopy from: {cartopy_url}')
    supports_mapping = False

matplotlib.use('Qt5Agg')

Qt = QtCore.Qt

MAKE_RE = re.compile('^(.*)\s+\([\d,]\)$')
SHAPEFILE = pathlib.Path(r'./Distribution/Areas.shp')
POSTCODEFILE = pathlib.Path(r'./uk-postcodes-master/postcodes.csv')

ALL_AUTOTRADER_COLOURS = (
    'Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Burgundy', 'Gold', 'Green', 'Grey', 'Magenta', 'Maroon',
    'Multicolour', 'Navy', 'Orange', 'Pink', 'Purple', 'Red', 'Silver', 'Turquoise', 'White', 'Yellow'
)

metadata_keys = ('make', 'model', 'fuel', 'trim', 'colour')
choices = ('Mileage', 'Time since first seen', 'Car Colour')  # tbd

# df with postcode info
postcode_data = None

db_manager = database_manager.DatabaseManager3()


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    # Todo: Fix this
    # self, section, orientation, role=None
    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


def load_place_and_postcodes() -> pd.DataFrame:
    """Loads the postcode csv data into the module var for later access"""
    # from https://github.com/Gibbs/uk-postcodes
    if not POSTCODEFILE.is_file():
        return None

    data = pd.read_csv(POSTCODEFILE)
    data['short_postcode'] = data['postcode'].str.replace(r'\d+', '', regex=True)
    data['town'] = data['town'].str.lower()
    data['region'] = data['region'].str.lower()
    data['uk_region'] = data['uk_region'].str.lower()
    return data


@lru_cache()
def location_to_postcode(place_name: str) -> str:
    # quick hack to sort improper loading of geo data
    if postcode_data is None:
        return ''

    postcode_set = set()
    for header in ('town', 'region', 'uk_region'):
        filtered = postcode_data[postcode_data[header] == place_name]
        short_postcodes = set(filtered['short_postcode'].tolist())
        postcode_set = postcode_set.union(short_postcodes)

    if len(postcode_set) > 1:
        # logging.warning(f'{len(postcode_set)} postcodes issue for {place_name}')
        pass

    if len(postcode_set) == 0:
        # logging.warning(f'No postcodes for {place_name}')
        return ''

    return postcode_set.pop()


class Worker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    emitDataFrame = QtCore.Signal(pd.DataFrame)

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, params):
        self.params = params
        QtCore.QThread.__init__(self)

    # def __del__(self):
    #     self.wait()

    # A QThread is run by calling it's start() function, which calls this run()
    # function in it's own "thread".
    def run(self):

        # pass the params to the scraper
        # is a dict that contains at least
        # make, model, postcode, min_year, and max_year as keys
        res = autotrader_scraper.get_cars(**self.params, include_writeoff='exclude')

        # it is possible for the scraper to blow up, so handle
        if len(res) == 0:
            self.emitDataFrame.emit(pd.DataFrame())
            return

        # convert result to dataframe, post-process, and emit back to main thread
        df = pd.DataFrame(res)
        df['price_int'] = df.price.str.replace('[,£]', '', regex=True).astype(int)
        df['postcode'] = df.location.apply(location_to_postcode)

        # perform some more clean up
        df = df[df['year'].notnull()]
        df = df[df['price_int'].notnull()]
        df = df[df['mileage'].notnull()]
        df['year'] = df['year'].str.extract('(\d{4})').astype(int)  # extract year

        # augment with search args
        # df['make'] = self.params['make'].lower()
        # df['model'] = self.params['model'].lower()
        for extra_key in metadata_keys:
            if extra_key not in df.columns and extra_key in self.params:
                df[extra_key] = self.params[extra_key]

        db_manager.append_snapshot(df)

        # spit out the df to main thread
        # df_queue.put(df)
        self.emitDataFrame.emit(df)


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # instance variables
        self.car_df = None
        self.search_params = {}
        self.avail_cars_list = None

        self.layout = QtWidgets.QVBoxLayout(self)

        # self.menu_bar = QtWidgets.QMenuBar(self)
        #
        # file_menu = self.menu_bar.addMenu('File')
        # edit_menu = self.menu_bar.addMenu('Edit')
        # exit_action = QtWidgets.QWidgetAction(self)
        # exit_action.setText('Exit')
        # exit_action.triggered.connect(exit)
        # file_menu.addAction(exit_action)

        # self.layout.addWidget(self.menu_bar)

        self.text = QtWidgets.QLabel("Waiting")  # alignment=QtCore.Qt.AlignCenter
        self.layout.addWidget(self.text)

        # these will be set to the result of plotting to distinguish between sold and available cars when clicking
        # see here https://stackoverflow.com/questions/45621544/matplotlib-pick-event-from-multiple-series
        self.avail_scatter = None

        # variables used to store objects from the shapeloader
        self.records = None
        self.geoms = None

        # makes
        make_hbox = QtWidgets.QHBoxLayout()
        make_label = QtWidgets.QLabel()
        make_label.setText('Make:')
        make_hbox.addWidget(make_label)
        self.makes_box = QtWidgets.QLineEdit()
        self.makes_box.setText('Lexus')
        # self.makes_box.addItems(makes)
        # self.makes_box.setCurrentText('Lexus')
        # self.makes_box.currentTextChanged.connect(self.on_makes_box_changed)
        make_hbox.addWidget(self.makes_box)
        self.layout.addLayout(make_hbox)

        # car model
        model_hbox = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel()
        model_label.setText('Model:')
        model_hbox.addWidget(model_label)
        self.model_box = QtWidgets.QLineEdit()
        self.model_box.setText('Rc 300h')
        # self.makes_box.currentTextChanged.connect(self.on_makes_box_changed)
        model_hbox.addWidget(self.model_box)
        self.layout.addLayout(model_hbox)

        # from year
        hbox = QtWidgets.QHBoxLayout()
        from_year_text = QtWidgets.QLabel()
        from_year_text.setText('From Year')
        hbox.addWidget(from_year_text)
        self.from_year = QtWidgets.QSpinBox()
        current_year = datetime.now().year
        self.from_year.setMaximum(current_year)
        self.from_year.setValue(current_year - 10)
        hbox.addWidget(self.from_year)
        self.layout.addLayout(hbox)

        # to year
        hbox2 = QtWidgets.QHBoxLayout()
        to_year_text = QtWidgets.QLabel()
        to_year_text.setText('To Year')
        hbox2.addWidget(to_year_text)
        self.to_year = QtWidgets.QSpinBox()
        self.to_year.setMaximum(current_year)
        self.to_year.setValue(current_year)
        hbox2.addWidget(self.to_year)
        self.layout.addLayout(hbox2)

        # tick box to show extra fields
        self.show_extra_fields_cb = QtWidgets.QCheckBox()
        self.show_extra_fields_cb.setChecked(True)
        self.show_extra_fields_cb.setText('Show Extra Data Fields (Fuel Type, Trim etc)')
        self.show_extra_fields_cb.stateChanged.connect(self.show_extra_fields_cb_state_changed)
        self.layout.addWidget(self.show_extra_fields_cb)

        # fuel type
        fuel_type_hbox = QtWidgets.QHBoxLayout()
        self.fuel_type_label = QtWidgets.QLabel()
        self.fuel_type_label.setText('Fuel Type')
        fuel_type_hbox.addWidget(self.fuel_type_label)
        self.fuel_type = QtWidgets.QLineEdit()
        fuel_type_hbox.addWidget(self.fuel_type)
        self.layout.addLayout(fuel_type_hbox)

        # trim
        trim_hbox = QtWidgets.QHBoxLayout()
        self.trim_label = QtWidgets.QLabel()
        self.trim_label.setText('Trim')
        trim_hbox.addWidget(self.trim_label)
        self.trim = QtWidgets.QLineEdit()
        self.trim.setText('F Sport')
        trim_hbox.addWidget(self.trim)
        self.layout.addLayout(trim_hbox)

        # colour
        colour_hbox = QtWidgets.QHBoxLayout()
        self.colour_label = QtWidgets.QLabel()
        self.colour_label.setText('Colour')
        colour_hbox.addWidget(self.colour_label)
        self.colour = QtWidgets.QLineEdit()
        colour_hbox.addWidget(self.colour)
        self.get_colour_info = QtWidgets.QPushButton()
        self.get_colour_info.setText('Get Colour Info')
        colour_hbox.addWidget(self.get_colour_info)
        self.get_colour_info.clicked.connect(self.on_get_colour_info_clicked)
        self.layout.addLayout(colour_hbox)

        self.hideable_controls = [
            self.fuel_type_label, self.fuel_type,
            self.trim_label, self.trim,
            self.colour_label, self.colour, self.get_colour_info
        ]

        # auto show/hide fields on startup
        show_controls = self.show_extra_fields_cb.checkState() == Qt.CheckState.Checked
        self.show_or_hide_extra_fields(show=show_controls)

        # button to get data and progress bar
        hbox3 = QtWidgets.QHBoxLayout()
        self.get_data = QtWidgets.QPushButton()
        self.get_data.setText('Get data')
        self.get_data.clicked.connect(self.on_get_data_clicked)
        hbox3.addWidget(self.get_data)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(1)
        self.progress_bar.setVisible(False)
        hbox3.addWidget(self.progress_bar)

        self.layout.addLayout(hbox3)

        self.table = QtWidgets.QTableView()
        self.layout.addWidget(self.table)

        self.send_to_map_button = QtWidgets.QPushButton()
        self.send_to_map_button.setText('Map')
        self.send_to_map_button.clicked.connect(self.on_send_to_map_button_clicked)

        if supports_mapping:
            self.layout.addWidget(self.send_to_map_button)

        self.graph_colour_choice = QtWidgets.QComboBox()
        self.graph_colour_choice.addItems(choices)
        self.layout.addWidget(self.graph_colour_choice)

        self.per_year_data_button = QtWidgets.QPushButton()
        self.per_year_data_button.setText('Show Price By Year')
        self.per_year_data_button.clicked.connect(self.per_year_data_button_clicked)
        self.layout.addWidget(self.per_year_data_button)

        self.edge_colour_is_car_colour_checkbox = QtWidgets.QCheckBox()
        self.edge_colour_is_car_colour_checkbox.setText('Set Marker Edge Colour To Car Colour')
        # self.edge_colour_is_car_colour_checkbox.setChecked(True)
        # self.edge_colour_is_car_colour_checkbox.stateChanged.connect(self.show_extra_fields_cb_state_changed)
        self.layout.addWidget(self.edge_colour_is_car_colour_checkbox)

        self.compare_cars_button = QtWidgets.QPushButton()
        self.compare_cars_button.setText('Show Mean Price Difference Between')
        self.compare_cars_button.clicked.connect(self.compare_cars_button_clicked)
        self.layout.addWidget(self.compare_cars_button)

        # self.main_menu = QtWidgets.QMenu()
        # self.main_menu.addAction('test')
        # self.bar = QtWidgets.QMenuBar(None)
        # self.bar.addMenu(self.main_menu)
        # menubar = self.men

        self.worker = None
        # self.db_manager = database_manager.DatabaseManager()

    def test(self):
        self.text.setText('Key!')

    @QtCore.Slot()
    def compare_cars_button_clicked(self):

        # get data
        rc300_df = pd.DataFrame(db_manager.fetch('Lexus', 'RC 300h'))
        is300_df = pd.DataFrame(db_manager.fetch('Lexus', 'IS 300'))

        # initialise plotting
        f, ax = plt.subplots()

        columns_tup = ['year', 'price_int']
        rc300_year_price_df = rc300_df[columns_tup]
        is300_year_price_df = is300_df[columns_tup]

        # compute mean
        rc_300_mean_price_df = rc300_year_price_df.groupby(['year']).mean()
        is300_mean_price_df = is300_year_price_df.groupby(['year']).mean()

        diff = rc_300_mean_price_df - is300_mean_price_df

        # plot
        ax.plot(rc_300_mean_price_df, figure=f, label='RC300h')
        ax.plot(is300_mean_price_df, figure=f, label='IS300')

        ax2 = ax.twinx()
        ax2.plot(diff, figure=f, label='Difference', linestyle='--')

        # configure graph and show
        plt.legend()
        ax.invert_xaxis()

        plt.xlabel('Year')
        plt.ylabel('Price (£)')

        plt.show()

    def show_or_hide_extra_fields(self, show=False):
        for control in self.hideable_controls:
            control.setVisible(show)
        self.get_colour_info.setVisible(False)  # todo: show when fixed

    @QtCore.Slot()
    def on_get_colour_info_clicked(self):
        for colour in ALL_AUTOTRADER_COLOURS:
            self.colour.setText(colour)
            # todo: goes waaaaaaaaaaaaaaaay to quick
            self.fetch_data()

    @QtCore.Slot()
    def show_extra_fields_cb_state_changed(self):
        show = self.show_extra_fields_cb.checkState()
        self.show_or_hide_extra_fields(show)

    @QtCore.Slot()
    def per_year_data_button_clicked(self):
        if type(self.car_df) is pd.DataFrame and not self.car_df.empty:

            # fetch saved cars of this make/model
            # note: this currently includes all cars, including the most recent ones
            # this is because they are added to the db as soon as they are fetched
            fetch_args = {k: v for k, v in self.search_params.items() if k in metadata_keys}
            stored_df = pd.DataFrame(db_manager.fetch(**fetch_args))
            if stored_df.empty:
                return

            # count how many cars were dropped in the pruning
            pre_drop_count = len(stored_df)

            # use these to bound the graph by filtering loaded df
            search_min_year = self.search_params['min_year']
            search_max_year = self.search_params['max_year']

            # keep certain cols
            # some are used for plotting, other because the df will be stored and used for the `pick` event
            cols_to_keep = ['year', 'price_int', 'mileage', 'days_old', 'link', 'hash']

            # get the users colour choice and decide if we need colour data, if not then delete it
            graph_colour_choice = self.graph_colour_choice.currentText()
            needs_colour_data = \
                graph_colour_choice == 'Car Colour'

            # if colour data is needed, add to the cols to keep and check we actually have something to plot
            if needs_colour_data:
                cols_to_keep.append('colour')
                if 'colour' not in stored_df.columns:
                    self.text.setText('Scrape cars with a colour option to get colour data')
                    return

                # if all good then covert to xkcd colours (which has many named colours), then filter to ones that exist
                stored_df['colour'] = 'xkcd:' + stored_df['colour'].str.lower()
                stored_df = stored_df[stored_df.colour.isin(mcd.XKCD_COLORS)]

            # filter to searched years
            stored_df = stored_df[(stored_df['year'] >= search_min_year) & (stored_df['year'] <= search_max_year)]

            # get min/max years from df (as might be different to the search range), use to generate xticks
            # sometimes returns float so cast to int
            df_min_year = int(min(stored_df['year']))
            df_max_year = int(max(stored_df['year']))
            r = tuple(range(df_max_year, df_min_year - 1, -1))

            # compute days old
            stored_df['datetime'] = stored_df['first_seen'].apply(datetime.fromtimestamp)
            max_datetime = max(stored_df['datetime'])
            stored_df['days_old'] = stored_df['datetime'].apply(lambda x: (max_datetime - x).days)

            # prune cols and rows with nan fields and extraneous cols
            stored_df = stored_df[cols_to_keep].dropna(how='any', axis=0)

            # generate filter to split df into sold and available cars
            # note: assume sold to mean exists in the database but not in the scrape
            current_car_hashes = set(self.car_df['hash'])
            stored_car_in_current_search = stored_df['hash'].isin(current_car_hashes)

            # report how many cars were dropped
            post_drop_count = len(stored_df)
            total_dropped = pre_drop_count - post_drop_count
            if total_dropped:
                logging.warning(f'Dropped {total_dropped} cars while gathering chart data')

            # create separate dfs for the two plotting axes
            avail_cars = pd.DataFrame(stored_df[stored_car_in_current_search])
            sold_cars = pd.DataFrame(stored_df[~stored_car_in_current_search])
            self.avail_cars_list = avail_cars['link'].to_list()

            # create a graph and draw scatter plots
            f2 = plt.figure()
            avail_scatter = partial(plt.scatter, x=avail_cars['year'], y=avail_cars['price_int'], marker='o',
                                    picker=True, edgecolors='black')
            sold_scatter = partial(plt.scatter, x=sold_cars['year'], y=sold_cars['price_int'], marker='x')

            # set up the rest of the partial plot configuration
            if graph_colour_choice == 'Mileage':
                avail_scatter = partial(avail_scatter, c=avail_cars['mileage'], cmap='RdYlGn_r')
                sold_scatter = partial(sold_scatter, c=sold_cars['mileage'], cmap='RdYlGn_r')
            elif graph_colour_choice == 'Time since first seen':
                avail_scatter = partial(avail_scatter, c=avail_cars['days_old'], cmap='RdYlGn_r')
                sold_scatter = partial(sold_scatter, c=sold_cars['days_old'], cmap='RdYlGn_r')
            elif graph_colour_choice == 'Car Colour':
                avail_scatter = partial(avail_scatter, c=avail_cars['colour'])
                sold_scatter = partial(sold_scatter, c=sold_cars['colour'])

            # fix a bug where plotting an empty dataframe caused the colourmap to plot between 0 and 1
            if not sold_cars.empty:
                sold_scatter()
            self.avail_scatter = avail_scatter()

            # Set graph labels, ticks, and other visual elements
            ax = f2.get_axes()[0]
            ax.set_xticks(r)
            ax.invert_xaxis()
            ax.set_xlabel('Year')
            ax.set_ylabel('Price (£)')
            ax.set_title(f'{self.search_params["model"]} Price Per Year')
            ax.grid(axis='y', linestyle='--')

            # only show the colourbar if we need it
            if graph_colour_choice != 'Car Colour':
                cbar = plt.colorbar()
                cbar.set_label(graph_colour_choice)

            # # I'm too dumb to interpret the data from this :(
            # f3 = plt.figure()
            # ax3d = f3.add_subplot(projection='3d')
            # ax3d.scatter(year_price_df['year'], year_price_df['mileage'], year_price_df['price_int'])
            # ax3d.set_xlabel('Year')
            # ax3d.set_ylabel('Miles')
            # ax3d.set_zlabel('Price')

            # bind the event for when someone clicks on the graph
            f2.canvas.mpl_connect("pick_event", self.on_pick)

            plt.show()

    def on_pick(self, event):

        # see if:
        # - user clicked an available car
        # - user clicked with left mouse button
        # if not then do nothing
        if event.artist is not self.avail_scatter or event.mouseevent.button != 1:  # 1 == left%
            return

        # otherwise open a browser window to the advert
        idx = list(event.ind).pop()
        link = self.avail_cars_list[idx]
        logging.debug(f'Clicked link: {link}')
        webbrowser.open(link)

    def fetch_data(self):
        self.get_data.setEnabled(False)
        self.table.reset()

        # show the "progress" bar
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(True)

        # invoke the worker to go and get the data from Autotrader, hook up the signal to handle it finishing, and run!
        params = {
            # 'make': self.makes_box.currentText(),  # when it was a combobox
            'make': self.makes_box.text(),
            'model': self.model_box.text(),
            'postcode': 'GL50 1EN',
            'min_year': self.from_year.value(),
            'max_year': self.to_year.value(),
        }

        # if the extra fields box is ticked grab all the extra info and slap it in
        if self.show_extra_fields_cb.checkState():
            if self.fuel_type.text():
                params['fuel'] = self.fuel_type.text()
            if self.trim.text():
                params['trim'] = self.trim.text()
            if self.colour.text():
                params['colour'] = self.colour.text()

        self.search_params = params
        self.worker = Worker(params)
        self.worker.emitDataFrame.connect(self.update_table, Qt.QueuedConnection)
        self.worker.start()

    @QtCore.Slot()
    def on_get_data_clicked(self):
        self.fetch_data()

    @QtCore.Slot()
    def update_table(self, dataframe):

        # kill the worker thread
        self.worker.terminate()  # fix threading termination
        self.worker.quit()
        self.worker = None

        # reset controls regardless of returned state
        self.progress_bar.setVisible(False)
        self.get_data.setEnabled(True)

        # allocate `car_df`
        self.car_df = dataframe

        # check to see if we got no data (e.g. 404 error from the scraper)
        # if dataframe.empty:
        if self.car_df.empty:
            self.table.reset()
            self.text.setText('Scrape failed, please retry')
            return

        # db_manager.append_snapshot(self.car_df)  # Hack to allow debugging

        # note: One interesting thing about the returned data is the AutoTrader link which I presume is unique
        # as such can use it as a key in a db. However, given we don't know if AutoTrader will enforce uniqueness
        # perhaps it would be prudent to add some other data to the key as well, such as make + model.

        # if we got data then display it
        model = pandasModel(self.car_df)
        self.table.setModel(model)
        self.text.setText(f'Got {len(self.car_df)} cars')

    @QtCore.Slot()
    def on_send_to_map_button_clicked(self):
        # ensure that there is something to map
        if type(self.car_df) is not pd.DataFrame:
            self.text.setText('No cars to map!')
            return

        self.send_to_map_button.setText(f'Mapping {len(self.car_df)} cars!')

        # get the mean price for each postcode, compute the max, and then the relevant percentages for each
        grouped_df = self.car_df.groupby(['postcode'])['postcode', 'price_int'].mean()
        max_price = grouped_df['price_int'].max()
        grouped_df['pct'] = grouped_df['price_int'] / max_price

        # please download from https://www.opendoorlogistics.com/downloads/
        # grab the data from the "Areas" data
        # n.b. "Area" in this case means first part of postcode e.g. BS, GL
        if not SHAPEFILE.is_file():
            return

        if self.records is None or self.geoms is None:
            logging.debug('Loading records and geoms')
            data = shpreader.Reader(str(SHAPEFILE))
            self.records = list(data.records())
            self.geoms = list(data.geometries())
        logging.debug(f'{len(self.records)} map geometries loaded')

        # create a map axis
        f = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())

        # iterate over all read areas and plot
        # if the Area is one we have info about colour it red
        # otherwise colour it white
        for geom, record in zip(self.geoms, self.records):
            postcode = record.attributes['name']
            # `facecolour` is a 3-tuple with each element in the range 0-1
            facecolour = (grouped_df.loc[postcode].pct, 0, 0) if postcode in grouped_df.index else 'white'
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=facecolour, edgecolor='black')

        # focus on UK
        ax.set_extent([-9, 2, 60, 49], ccrs.PlateCarree())
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # load data at program start

    if supports_mapping:
        postcode_data = load_place_and_postcodes()
        if postcode_data is None:
            logging.error('Please download postcode data from https://github.com/Gibbs/uk-postcodes')
            supports_mapping = False

        if not SHAPEFILE.is_file():
            logging.error('Please download shapefile data from https://www.opendoorlogistics.com/downloads/')
            supports_mapping = False

    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
