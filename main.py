# Todo:
# Fix charts randomly failing to draw - might be related to threading issues, has been better since removing printing
# Fetch from db in worker
# Add other scrape options - some done
# Add "new car" markers to graph - need to rethink the saving strategy to be able to do this
# Add start search on enter key


import logging
import sys
from typing import Dict, Union
import webbrowser
from datetime import datetime
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PySide2 import QtCore, QtWidgets  # Downgraded to PySide2 (Qt5) for proper matplotlib compatibility
from PySide2.QtCore import QAbstractTableModel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import numpy as np

import database_manager
import scraper2 as autotrader_scraper

# set up a module logger
LOGGER = logging.getLogger(__name__)


matplotlib.use('Qt5Agg')

Qt = QtCore.Qt

METADATA_KEYS = ('make', 'model', 'fuel', 'trim', 'colour')
PLOT_MARKER_CHOICES = ('Mileage', 'Time since first seen', 'Score') 

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

class Worker(QtCore.QThread):
    # This is the signal that will be emitted during the processing.
    # By including int as an argument, it lets the signal know to expect
    # an integer argument when emitting.
    emitDataFrame = QtCore.Signal(pd.DataFrame)

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self, params: Dict[str, Union[str, int]]):
        self.params = params
        QtCore.QThread.__init__(self)

    # def __del__(self):
    #     self.wait()

    # A QThread is run by calling it's start() function, which calls this run() function
    def run(self):

        LOGGER.debug('Beginning worker run()')

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

        # filter out rows without key data fields set
        df = df[df['year'].notnull()]
        df = df[df['price'].notnull()]
        df = df[df['mileage'].notnull()]

        # augment with search args
        for extra_key in METADATA_KEYS:
            if extra_key not in df.columns and extra_key in self.params:
                df[extra_key] = self.params[extra_key]

        db_manager.append_snapshot(df)

        # spit out the df to main thread
        self.emitDataFrame.emit(df)


class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # instance variables
        self.car_df = None
        self.search_params = {}
        self.avail_cars_list = None

        self.requires_db_reload = True
        '''Set to True when the widget should reload entries from the database'''

        self.layout = QtWidgets.QVBoxLayout(self)

        self.text = QtWidgets.QLabel("Waiting")
        self.text.setMaximumHeight(25)
        self.layout.addWidget(self.text)

        # these will be set to the result of plotting to distinguish between sold and available cars when clicking
        # see here https://stackoverflow.com/questions/45621544/matplotlib-pick-event-from-multiple-series
        self.avail_scatter = None

        # makes
        make_hbox = QtWidgets.QHBoxLayout()
        make_label = QtWidgets.QLabel()
        make_label.setText('Make:')
        make_hbox.addWidget(make_label)
        self.makes_box = QtWidgets.QLineEdit()
        self.makes_box.setText('Lexus')
        make_hbox.addWidget(self.makes_box)
        self.layout.addLayout(make_hbox)

        # car model
        model_hbox = QtWidgets.QHBoxLayout()
        model_label = QtWidgets.QLabel()
        model_label.setText('Model:')
        model_hbox.addWidget(model_label)
        self.model_box = QtWidgets.QLineEdit()
        self.model_box.setText('IS 300')
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

        # max miles
        miles_hbox = QtWidgets.QHBoxLayout()
        self.max_miles_label = QtWidgets.QLabel()
        self.max_miles_label.setText('Max Miles (0 for unlimited)')
        miles_hbox.addWidget(self.max_miles_label)
        self.max_miles = QtWidgets.QSpinBox()
        self.max_miles.setMaximum(2**31-1)  # this is the maximum supported number
        self.max_miles.setValue(50000)
        miles_hbox.addWidget(self.max_miles)
        self.layout.addLayout(miles_hbox)

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
        self.layout.addLayout(colour_hbox)

        self.hideable_controls = [
            self.fuel_type_label, self.fuel_type,
            self.trim_label, self.trim,
            self.max_miles_label, self.max_miles
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
        self.table.setVisible(False)  # TODO
        self.layout.addWidget(self.table)

        self.graph_colour_choice = QtWidgets.QComboBox()
        self.graph_colour_choice.addItems(PLOT_MARKER_CHOICES)
        self.graph_colour_choice.currentTextChanged.connect(self.update_graph)
        self.layout.addWidget(self.graph_colour_choice)

        self.per_year_data_button = QtWidgets.QPushButton()
        self.per_year_data_button.setText('Show Price By Year')
        self.per_year_data_button.clicked.connect(self.update_graph)
        self.layout.addWidget(self.per_year_data_button)

        self.edge_colour_is_car_colour_checkbox = QtWidgets.QCheckBox()
        self.edge_colour_is_car_colour_checkbox.setText('Set Marker Edge Colour To Car Colour')
        # self.edge_colour_is_car_colour_checkbox.setChecked(True)
        # self.edge_colour_is_car_colour_checkbox.stateChanged.connect(self.show_extra_fields_cb_state_changed)
        self.layout.addWidget(self.edge_colour_is_car_colour_checkbox)

        self.new_cars_only_checkbox = QtWidgets.QCheckBox()
        self.new_cars_only_checkbox.setText('Only Show Cars Added Today')
        self.new_cars_only_checkbox.stateChanged.connect(self.update_graph)
        self.layout.addWidget(self.new_cars_only_checkbox)

        # for drawing graph in the main window
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)

        # bind the event for when someone clicks on the graph and render
        self.figure.canvas.mpl_connect("pick_event", self.on_pick)

        self.worker = None

    def show_or_hide_extra_fields(self, show: bool) -> None:
        for control in self.hideable_controls:
            control.setVisible(show)

    @QtCore.Slot()
    def show_extra_fields_cb_state_changed(self):
        show = self.show_extra_fields_cb.checkState()
        self.show_or_hide_extra_fields(show)

    @QtCore.Slot()
    def update_graph(self):

        # type check
        if not type(self.car_df) is pd.DataFrame or self.car_df.empty:
            return

        # fetch saved cars of this make/model
        # note: this currently includes all cars, including the most recent ones
        # this is because they are added to the db as soon as they are fetched
        if self.requires_db_reload:
            LOGGER.debug('Loading additional cars from database')
            fetch_args = {k: v for k, v in self.search_params.items() if k in METADATA_KEYS}
            self.stored_df = pd.DataFrame(db_manager.fetch(**fetch_args))
            if self.stored_df.empty:
                return
                
            # compute days old
            self.stored_df['datetime'] = self.stored_df['first_seen'].apply(datetime.fromtimestamp)
            max_datetime = max(self.stored_df['datetime'])
            self.stored_df['days_old'] = self.stored_df['datetime'].apply(lambda x: (max_datetime - x).days)

        self.requires_db_reload = False

        # count how many cars were dropped in the pruning
        pre_drop_count = len(self.stored_df)

        # use these to bound the graph by filtering loaded df
        search_min_year = self.search_params['min_year']
        search_max_year = self.search_params['max_year']
        year_range = tuple(range(search_min_year, search_max_year + 1))

        # keep certain cols
        # some are used for plotting, other because the df will be stored and used for the `pick` event
        cols_to_keep = ['year', 'price', 'mileage', 'days_old', 'url', 'hash']
        
        # filter to searched years
        stored_df = self.stored_df[(self.stored_df['year'] >= search_min_year) & (self.stored_df['year'] <= search_max_year)]

        mean_price = stored_df.groupby(['year'])['price'].mean()

        # filter to cars added today only if asked to
        if self.new_cars_only_checkbox.checkState() == Qt.CheckState.Checked:
            stored_df = stored_df[self.stored_df.days_old == 0]

        # prune cols and rows with nan fields and extraneous cols
        stored_df = stored_df[cols_to_keep].dropna(how='any', axis=0)

        # generate filter to split df into sold and available cars
        # note: assume sold to mean exists in the database but not in the scrape
        current_car_hashes = set(self.car_df['hash'])
        stored_car_in_current_search = stored_df['hash'].isin(current_car_hashes)

        # report how many cars were dropped
        post_drop_count = len(stored_df)
        total_dropped = pre_drop_count - post_drop_count
        if total_dropped > 0:
            LOGGER.warning('Dropped %d cars while gathering chart data', total_dropped)

        # create separate dfs available and sold cars because the scatter plot
        # only takes separate markers per plot, not per datapoint
        avail_cars = pd.DataFrame(stored_df[stored_car_in_current_search])
        sold_cars = pd.DataFrame(stored_df[~stored_car_in_current_search])
        self.avail_cars_list = avail_cars['url'].to_list()

        # create a graph and draw scatter plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        avail_scatter = partial(ax.scatter, x=avail_cars['year'], y=avail_cars['price'], marker='o',
                                picker=True, edgecolors='black')
        sold_scatter = partial(ax.scatter, x=sold_cars['year'], y=sold_cars['price'], marker='x')

        # compute means and standard deviations (stds) per year and append those to avail_cars
        means_df = avail_cars.groupby(['year']).mean()
        stds_df = avail_cars.groupby(['year']).std()
        avail_cars = avail_cars.join(means_df, on='year', how='left', rsuffix='_grp_mean')
        avail_cars = avail_cars.join(stds_df, on='year', how='left', rsuffix='_grp_std')

        # compute per-year z-score of miles and price
        avail_cars['scaled_price'] = (avail_cars['price'] - avail_cars['price_grp_mean']) / avail_cars['price_grp_std']
        avail_cars['scaled_miles'] = (avail_cars['mileage'] - avail_cars['mileage_grp_mean']) / avail_cars['mileage_grp_std']

        # create a new column that combines other features that is a good indicator of a bargain
        avail_cars['score'] = avail_cars['scaled_price'] + avail_cars['scaled_miles']

        # set up the rest of the partial plot configuration
        graph_colour_choice = self.graph_colour_choice.currentText()
        if graph_colour_choice == 'Mileage':
            avail_scatter = partial(avail_scatter, c=avail_cars['mileage'], cmap='RdYlGn_r')
            sold_scatter = partial(sold_scatter, c=sold_cars['mileage'], cmap='RdYlGn_r')
        elif graph_colour_choice == 'Time since first seen':
            avail_scatter = partial(avail_scatter, c=avail_cars['days_old'], cmap='RdYlGn_r')
            sold_scatter = partial(sold_scatter, c=sold_cars['days_old'], cmap='RdYlGn_r')
        elif graph_colour_choice == 'Score':
            avail_scatter = partial(avail_scatter, c=avail_cars['score'], cmap='RdYlGn_r')
            sold_scatter = partial(sold_scatter, c='black', edgecolors='black')

        # fix a bug where plotting an empty dataframe caused the colourmap to plot between 0 and 1
        if not sold_cars.empty:
            sold_scatter()
        self.avail_scatter = avail_scatter()

        # compute mean price of the cars per year and plot
        ax.plot(mean_price)

        # Set graph labels, ticks, and other visual elements
        ax.set_xticks(year_range)
        ax.invert_xaxis()
        ax.set_xlabel('Year')
        ax.set_ylabel('Price (£)')
        ax.set_title(f'{self.search_params["model"]} Price Per Year')
        ax.grid(axis='y', linestyle='--')

        # show the colourbar
        cbar = self.figure.colorbar(self.avail_scatter)
        cbar.set_label(graph_colour_choice)

        self.canvas.draw()

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
        LOGGER.debug(f'Clicked link: {link}')
        webbrowser.open(link)

    def scrape_data(self):
        ''' Gathers the params from the form and passes them to the scraper.
        Converts the returned list of dicts into a dataframe, which it emits.
        '''
        self.get_data.setEnabled(False)
        self.table.reset()

        # show the "progress" bar
        self.progress_bar.setMaximum(0)
        self.progress_bar.setVisible(True)

        # invoke the worker to go and get the data from Autotrader, hook up the signal to handle it finishing, and run!
        params = {
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
            if self.max_miles.value() > 0:
                params['max_miles'] = self.max_miles.value()

        self.search_params = params
        self.worker = Worker(params)
        self.worker.emitDataFrame.connect(self.update_table, Qt.QueuedConnection)
        self.worker.start()

    @QtCore.Slot()
    def on_get_data_clicked(self):
        self.requires_db_reload = True
        self.scrape_data()

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

        # note: One interesting thing about the returned data is the AutoTrader link which I presume is unique
        # as such can use it as a key in a db. However, given we don't know if AutoTrader will enforce uniqueness
        # perhaps it would be prudent to add some other data to the key as well, such as make + model.

        # if we got data then display it
        # model = pandasModel(self.car_df)
        # self.table.setModel(model)
        # self.text.setText(f'Got {len(self.car_df)} cars')

        self.update_graph()

class LoggingDenoiser(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return 'matplotlib' not in record.getMessage()

if __name__ == "__main__":
    logging.basicConfig(level='DEBUG')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
