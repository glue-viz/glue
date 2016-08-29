from __future__ import absolute_import, division, print_function

import json

import mock
import pytest
from mock import patch

plotly = pytest.importorskip('plotly')
from plotly.exceptions import PlotlyError

from glue.tests.helpers import requires_plotly
from glue.core import Data, DataCollection
from glue.app.qt import GlueApplication
from glue.viewers.histogram.qt import HistogramWidget

from glue.plugins.exporters.plotly.export_plotly import build_plotly_call

from .. import QtPlotlyExporter

plotly_sign_in = mock.MagicMock()
plotly_plot = mock.MagicMock()


SIGN_IN_ERROR = """
Aw, snap! You tried to use our API as the user 'BATMAN', but
the supplied API key doesn't match our records. You can view
your API key at plot.ly/settings.
"""

MAX_PRIVATE_ERROR = """
This file cannot be saved as private, your current Plotly account has
filled its quota of private files. You can still save public files, or you can
upgrade your account to save more files privately by visiting your account at
https://plot.ly/settings/subscription. To make a file public in the API, set
the optional argument 'world_readable' to true.
"""


def make_credentials_file(path, username='', api_key=''):
    credentials = {}
    credentials['username'] = username
    credentials['api_key'] = api_key
    credentials['proxy_username'] = ''
    credentials['proxy_password'] = ''
    credentials['stream_ids'] = []
    with open(path, 'w') as f:
        json.dump(credentials, f)
    plotly.files.FILE_CONTENT[path] = credentials


@requires_plotly
class TestQtPlotlyExporter():

    def setup_class(self):

        data = Data(x=[1, 2, 3], y=[2, 3, 4], label='data')
        dc = DataCollection([data])
        app = GlueApplication(dc)

        data.style.color = '#000000'
        v = app.new_data_viewer(HistogramWidget, data=data)
        v.component = data.id['y']
        v.xmin = 0
        v.xmax = 10
        v.bins = 20

        self.args, self.kwargs = build_plotly_call(app)

    def get_exporter(self):
        return QtPlotlyExporter(plotly_args=self.args, plotly_kwargs=self.kwargs)

    def test_default(self, tmpdir):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file)

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):

            exporter = self.get_exporter()

            assert exporter.radio_account_glue.isChecked()
            assert exporter.radio_sharing_public.isChecked()
            assert not exporter.radio_sharing_secret.isEnabled()
            assert not exporter.radio_sharing_private.isEnabled()

    def test_default_with_credentials(self, tmpdir):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file, username='batman', api_key='batmobile')

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):

            exporter = self.get_exporter()

            assert exporter.radio_account_config.isChecked()
            assert 'username: batman' in exporter.radio_account_config.text()
            assert exporter.radio_sharing_secret.isChecked()
            assert exporter.radio_sharing_secret.isEnabled()
            assert exporter.radio_sharing_private.isEnabled()

    def test_toggle_account_sharing(self, tmpdir):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file)

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):

            exporter = self.get_exporter()

            assert not exporter.radio_sharing_secret.isEnabled()
            assert not exporter.radio_sharing_private.isEnabled()

            exporter.radio_account_manual.setChecked(True)

            assert exporter.radio_sharing_secret.isEnabled()
            assert exporter.radio_sharing_private.isEnabled()

            exporter.radio_account_glue.setChecked(True)

            assert not exporter.radio_sharing_secret.isEnabled()
            assert not exporter.radio_sharing_private.isEnabled()

    def test_edit_username_toggle_custom(self, tmpdir):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file)

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):

            exporter = self.get_exporter()

            assert exporter.radio_account_glue.isChecked()
            assert not exporter.radio_account_manual.isChecked()
            exporter.username = 'a'
            assert not exporter.radio_account_glue.isChecked()
            assert exporter.radio_account_manual.isChecked()

            exporter.radio_account_glue.setChecked(True)

            assert exporter.radio_account_glue.isChecked()
            assert not exporter.radio_account_manual.isChecked()
            exporter.api_key = 'a'
            assert not exporter.radio_account_glue.isChecked()
            assert exporter.radio_account_manual.isChecked()

    def test_accept_default(self, tmpdir):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file)

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):
            with patch('plotly.plotly.plot', mock.MagicMock()):
                with patch('webbrowser.open_new_tab') as open_new_tab:
                    exporter = self.get_exporter()
                    exporter.accept()
                    assert exporter.text_status.text() == 'Exporting succeeded'

    ERRORS = [
        (PlotlyError(SIGN_IN_ERROR), 'Authentication with username batman failed'),
        (PlotlyError(MAX_PRIVATE_ERROR), 'Maximum number of private plots reached'),
        (PlotlyError('Oh noes!'), 'An unexpected error occurred'),
        (TypeError('A banana is not an apple'), 'An unexpected error occurred')
    ]

    @pytest.mark.parametrize(('error', 'status'), ERRORS)
    def test_accept_errors(self, tmpdir, error, status):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file, username='batman', api_key='batmobile')

        plot = mock.MagicMock(side_effect=error)

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):
            with patch('plotly.plotly.plot', plot):
                with patch('webbrowser.open_new_tab') as open_new_tab:
                    exporter = self.get_exporter()
                    exporter.accept()
                    assert exporter.text_status.text() == status

    @pytest.mark.parametrize(('error', 'status'), ERRORS)
    def test_fix_url(self, tmpdir, error, status):

        credentials_file = tmpdir.join('.credentials').strpath

        make_credentials_file(credentials_file, username='batman', api_key='batmobile')

        plot = mock.MagicMock(return_value='https://plot.ly/~batman/6?share_key=rbkWvJQn6cyj3HMMGROiqI')

        with patch('plotly.tools.CREDENTIALS_FILE', credentials_file):
            with patch('plotly.plotly.plot', plot):
                with patch('webbrowser.open_new_tab') as open_new_tab:
                    exporter = self.get_exporter()
                    exporter.accept()
                    assert open_new_tab.called_once_with('https://plot.ly/~batman/6/?share_key=rbkWvJQn6cyj3HMMGROiqI')
