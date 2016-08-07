import os
import sys
import traceback
import webbrowser

from qtpy import QtWidgets
from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue.utils.qt.widget_properties import TextProperty, ButtonProperty


class QtPlotlyExporter(QtWidgets.QDialog):

    save_settings = ButtonProperty('checkbox_save')
    username = TextProperty('text_username')
    api_key = TextProperty('text_api_key')
    title = TextProperty('text_title')
    legend = ButtonProperty('checkbox_legend')

    def __init__(self, plotly_args=[], plotly_kwargs={}, parent=None):

        super(QtPlotlyExporter, self).__init__(parent=parent)

        self.plotly_args = plotly_args
        self.plotly_kwargs = plotly_kwargs

        self.ui = load_ui('exporter.ui', self,
                          directory=os.path.dirname(__file__))

        self.button_cancel.clicked.connect(self.reject)
        self.button_export.clicked.connect(self.accept)

        # Set up radio button groups

        self._radio_account = QtWidgets.QButtonGroup()
        self._radio_account.addButton(self.ui.radio_account_glue)
        self._radio_account.addButton(self.ui.radio_account_config)
        self._radio_account.addButton(self.ui.radio_account_manual)

        self._radio_sharing = QtWidgets.QButtonGroup()
        self._radio_sharing.addButton(self.ui.radio_sharing_public)
        self._radio_sharing.addButton(self.ui.radio_sharing_secret)
        self._radio_sharing.addButton(self.ui.radio_sharing_private)

        # Find out stored credentials (note that this will create the
        # credentials file if it doesn't already exist)

        from plotly import plotly

        credentials = plotly.get_credentials()
        config_available = credentials['username'] != "" and credentials['api_key'] != ""

        if config_available:
            self.ui.radio_account_config.setChecked(True)
            label = self.ui.radio_account_config.text()
            self.ui.radio_account_config.setText(label + " (username: {0})".format(credentials['username']))
        else:
            self.ui.radio_account_glue.setChecked(True)
            self.ui.radio_account_config.setEnabled(False)

        self.ui.radio_sharing_secret.setChecked(True)

        self.ui.text_username.textChanged.connect(nonpartial(self._set_manual_mode))
        self.ui.text_api_key.textChanged.connect(nonpartial(self._set_manual_mode))
        self.ui.radio_account_glue.toggled.connect(nonpartial(self._set_allowed_sharing_modes))

        self.set_status('', color='black')

        self._set_allowed_sharing_modes()

    def _set_manual_mode(self):
        self.ui.radio_account_manual.setChecked(True)

    def _set_allowed_sharing_modes(self):
        if self.ui.radio_account_glue.isChecked():
            self.ui.radio_sharing_public.setChecked(True)
            self.ui.radio_sharing_secret.setEnabled(False)
            self.ui.radio_sharing_private.setEnabled(False)
        else:
            self.ui.radio_sharing_secret.setEnabled(True)
            self.ui.radio_sharing_private.setEnabled(True)
        QtWidgets.QApplication.instance().processEvents()

    def accept(self):

        # In future we might be able to use more fine-grained exceptions
        # https://github.com/plotly/plotly.py/issues/524

        self.set_status('Signing in and plotting...', color='blue')

        auth = {}

        if self.ui.radio_account_glue.isChecked():
            auth['username'] = 'Glue'
            auth['api_key'] = 't24aweai14'
        elif self.ui.radio_account_config.isChecked():
            auth['username'] = ''
            auth['api_key'] = ''
        else:
            if self.username == "":
                self.set_status("Username not set", color='red')
                return
            elif self.api_key == "":
                self.set_status("API key not set", color='red')
                return
            else:
                auth['username'] = self.username
                auth['api_key'] = self.api_key

        from plotly import plotly
        from plotly.exceptions import PlotlyError
        from plotly.tools import set_credentials_file

        # Signing in - at the moment this will not check the credentials so we
        # can't catch any issues until later, but I've opened an issue for this:
        # https://github.com/plotly/plotly.py/issues/525
        plotly.sign_in(auth['username'], auth['api_key'])

        if self.ui.radio_sharing_public.isChecked():
            self.plotly_kwargs['sharing'] = 'public'
        elif self.ui.radio_sharing_secret.isChecked():
            self.plotly_kwargs['sharing'] = 'secret'
        else:
            self.plotly_kwargs['sharing'] = 'private'

        # We need to fix URLs, so we can't let plotly open it yet
        # https://github.com/plotly/plotly.py/issues/526
        self.plotly_kwargs['auto_open'] = False

        # Get title and legend preferences from the window
        self.plotly_args[0]['layout']['showlegend'] = self.legend
        self.plotly_args[0]['layout']['title'] = self.title

        try:
            plotly_url = plotly.plot(*self.plotly_args, **self.plotly_kwargs)
        except PlotlyError as exc:
            print("Plotly exception:")
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            if "the supplied API key doesn't match our records" in exc.args[0]:
                username = auth['username'] or plotly.get_credentials()['username']
                self.set_status("Authentication with username {0} failed".format(username), color='red')
            elif "filled its quota of private files" in exc.args[0]:
                self.set_status("Maximum number of private plots reached", color='red')
            else:
                self.set_status("An unexpected error occurred", color='red')
            return
        except:
            print("Plotly exception:")
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            self.set_status("An unexpected error occurred", color='red')
            return

        self.set_status('Exporting succeeded', color='blue')

        if self.save_settings and self.ui.radio_account_manual.isChecked():
            try:
                set_credentials_file(**auth)
            except Exception:
                print("Plotly exception:")
                print('-' * 60)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60)
                self.set_status('Exporting succeeded (but saving login failed)', color='blue')

        # We need to fix URL
        # https://github.com/plotly/plotly.py/issues/526
        if self.plotly_kwargs['sharing'] == 'secret':
            pos = plotly_url.find('?share_key')
            if pos >= 0:
                if plotly_url[pos - 1] != '/':
                    plotly_url = plotly_url.replace('?share_key', '/?share_key')

        print("Plotly URL: {0}".format(plotly_url))

        webbrowser.open_new_tab(plotly_url)

        super(QtPlotlyExporter, self).accept()

    def set_status(self, text, color):
        self.ui.text_status.setText(text)
        self.ui.text_status.setStyleSheet("color: {0}".format(color))
        QtWidgets.QApplication.instance().processEvents()

