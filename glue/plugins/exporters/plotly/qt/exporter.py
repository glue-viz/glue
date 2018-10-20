from __future__ import absolute_import, division, print_function

import os
import sys
import logging
import traceback
import webbrowser

from qtpy import QtWidgets
from glue.utils import nonpartial
from glue.utils.qt import load_ui
from glue.utils.qt.widget_properties import TextProperty, ButtonProperty
from glue.plugins.exporters.plotly.export_plotly import build_plotly_call


def save_plotly(application):
    """
    Save a Glue session to a plotly plot

    This is currently restricted to 1-4 scatterplots or histograms

    Parameters
    ----------
    application : `~glue.core.application_base.Application`
        Glue application to save
    label : str
        Label for the exported plot
    """

    args, kwargs = build_plotly_call(application)

    logging.getLogger(__name__).debug(args, kwargs)

    # TODO: check what current GUI framework is

    exporter = QtPlotlyExporter(plotly_args=args, plotly_kwargs=kwargs)
    exporter.exec_()


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
            self.ui.radio_account_manual.setChecked(True)

        self.ui.radio_sharing_secret.setChecked(True)

        self.ui.text_username.textChanged.connect(nonpartial(self._set_manual_mode))
        self.ui.text_api_key.textChanged.connect(nonpartial(self._set_manual_mode))

        self.set_status('', color='black')

    def _set_manual_mode(self):
        self.ui.radio_account_manual.setChecked(True)

    def accept(self):

        # In future we might be able to use more fine-grained exceptions
        # https://github.com/plotly/plotly.py/issues/524

        self.set_status('Signing in and plotting...', color='blue')

        auth = {}

        if self.ui.radio_account_config.isChecked():
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
            plotly.sign_in(auth['username'], auth['api_key'])
            plotly_url = plotly.plot(*self.plotly_args, **self.plotly_kwargs)
        except PlotlyError as exc:
            print("Plotly exception:")
            print('-' * 60)
            traceback.print_exc(file=sys.stdout)
            print('-' * 60)
            if ('the supplied API key doesn\'t match our records' in exc.args[0] or
                    'Sign in failed' in exc.args[0]):
                username = auth['username'] or plotly.get_credentials()['username']
                self.set_status("Authentication failed".format(username), color='red')
            elif "filled its quota of private files" in exc.args[0]:
                self.set_status("Maximum number of private plots reached", color='red')
            elif "Accounts on the Community Plan cannot save private files" in exc.args[0]:
                self.set_status("Accounts on the Community Plan cannot save private files", color='red')
            else:
                self.set_status("An unexpected error occurred", color='red')
            return
        except Exception:
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

        print("Plotly URL: {0}".format(plotly_url))

        webbrowser.open_new_tab(plotly_url)

        super(QtPlotlyExporter, self).accept()

    def set_status(self, text, color):
        self.ui.text_status.setText(text)
        self.ui.text_status.setStyleSheet("color: {0}".format(color))
        QtWidgets.QApplication.instance().processEvents()
