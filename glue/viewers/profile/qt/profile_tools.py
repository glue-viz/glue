import os
import traceback

import numpy as np

from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui

from matplotlib.colors import to_hex

from glue.config import fit_plugin, viewer_tool
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.profile.mouse_mode import NavigateMouseMode, RangeMouseMode
from glue.viewers.profile.qt.fitters import FitSettingsWidget
from glue.utils.qt import Worker
from glue.viewers.common.qt.tool import Tool

__all__ = ['ProfileTools']


MODES = ['navigate', 'fit', 'collapse']


@viewer_tool
class ProfileTool(Tool):

    tool_id = 'profile-tools'

    def __init__(self, viewer):
        super(ProfileTool, self).__init__(viewer)
        self._profile_tools = ProfileTools(viewer)
        container_widget = QtWidgets.QSplitter(Qt.Horizontal)
        plot_widget = viewer.centralWidget()
        container_widget.addWidget(plot_widget)
        container_widget.addWidget(self._profile_tools)
        viewer.setCentralWidget(container_widget)
        # container_widget = QtWidgets.QWidget()
        # container_layout = QtWidgets.QHBoxLayout()
        # plot_widget = viewer.centralWidget()
        # container_widget.setLayout(container_layout)
        # container_layout.addWidget(plot_widget)
        # container_layout.addWidget(self._profile_tools)
        # viewer.setCentralWidget(container_widget)
        self._profile_tools.enable()
        self._profile_tools.hide()

    def activate(self):
        if self._profile_tools.isVisible():
            self._profile_tools.hide()
        else:
            self._profile_tools.show()


class ProfileTools(QtWidgets.QWidget):

    def __init__(self, parent=None):

        super(ProfileTools, self).__init__(parent=parent)

        self.ui = load_ui('profile_tools.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tabs)

        self.viewer = parent

    def enable(self):

        self.nav_mode = NavigateMouseMode(self.viewer)
        self.rng_mode = RangeMouseMode(self.viewer)

        self.ui.tabs.setCurrentIndex(0)

        self.ui.tabs.currentChanged.connect(self._on_tab_change)
        self._on_tab_change()

        self.ui.button_settings.clicked.connect(self._on_settings)
        self.ui.button_fit.clicked.connect(self._on_fit)
        self.ui.button_clear.clicked.connect(self._on_clear)
        self.ui.button_collapse.clicked.connect(self._on_collapse)

        font = QtGui.QFont("Courier")
        font.setStyleHint(font.Monospace)
        self.ui.text_log.document().setDefaultFont(font)
        self.ui.text_log.setLineWrapMode(self.ui.text_log.NoWrap)

        self.axes = self.viewer.axes
        self.canvas = self.axes.figure.canvas

        self._fit_artists = []

        for fitter in list(fit_plugin):
            self.ui.combosel_fit_function.addItem(fitter.label, userData=fitter())

        self._toolbar_connected = False

        self.viewer.toolbar_added.connect(self._on_toolbar_added)

    @property
    def fitter(self):
        # FIXME: might not work with PyQt4
        return self.ui.combosel_fit_function.currentData()

    def _on_settings(self):
        d = FitSettingsWidget(self.fitter)
        d.exec_()

    def _on_fit(self):
        """
        Fit a model to the data

        The fitting happens on a dedicated thread, to keep the UI
        responsive
        """

        x_range = self.rng_mode.state.x_range
        fitter = self.fitter

        def on_success(result):
            fit_results, x, y = result
            report = ""
            for layer_artist in fit_results:
                report += ("<b><font color='{0}'>{1}</font>"
                           "</b>".format(to_hex(layer_artist.state.color),
                                                    layer_artist.layer.label))
                report += "<pre>" + fitter.summarize(fit_results[layer_artist], x, y) + "</pre>"
            self._report_fit(report)
            self._plot_fit(fitter, fit_results, x, y)

        def on_fail(exc_info):
            exc = '\n'.join(traceback.format_exception(*exc_info))
            self._report_fit("Error during fitting:\n%s" % exc)

        def on_done():
            self.ui.button_fit.setText("Fit")
            self.ui.button_fit.setEnabled(True)
            self.canvas.draw()

        self.ui.button_fit.setText("Running...")
        self.ui.button_fit.setEnabled(False)

        w = Worker(self._fit, fitter, xlim=x_range)
        w.result.connect(on_success)
        w.error.connect(on_fail)
        w.finished.connect(on_done)

        self._fit_worker = w  # hold onto a reference
        w.start()

    def _report_fit(self, report):
        self.ui.text_log.document().setHtml(report)

    def _on_clear(self):
        self.ui.text_log.document().setPlainText('')
        self._clear_fit()
        self.canvas.draw()

    def _fit(self, fitter, xlim=None):

        # We cycle through all the visible layers and get the plotted data
        # for each one of them.

        results = {}
        for layer in self.viewer.layers:
            if layer.enabled and layer.visible:
                if hasattr(layer, '_visible_data'):
                    x, y = layer._visible_data
                    x = np.asarray(x)
                    y = np.asarray(y)
                    keep = (x >= min(xlim)) & (x <= max(xlim))
                    if len(x) > 0:
                        results[layer] = fitter.build_and_fit(x[keep], y[keep])

        return results, x, y

    def _clear_fit(self):
        for artist in self._fit_artists[:]:
            artist.remove()
            self._fit_artists.remove(artist)

    def _plot_fit(self, fitter, fit_result, x, y):

        self._clear_fit()

        for layer in fit_result:
            # y_model = fitter.predict(fit_result[layer], x)
            self._fit_artists.append(fitter.plot(fit_result[layer], self.axes, x,
                                                 alpha=layer.state.alpha,
                                                 linewidth=layer.state.linewidth * 0.5,
                                                 color=layer.state.color)[0])

        self.canvas.draw()

    def _on_collapse(self):
        pass

    @property
    def mode(self):
        return MODES[self.tabs.currentIndex()]

    def _on_toolbar_added(self, *event):
        self.viewer.toolbar.tool_activated.connect(self._on_toolbar_activate)
        self.viewer.toolbar.tool_deactivated.connect(self._on_tab_change)

    def _on_toolbar_activate(self, *event):
        self.rng_mode.deactivate()
        self.nav_mode.deactivate()

    def _on_tab_change(self, *event):
        mode = self.mode
        if mode == 'navigate':
            self.rng_mode.deactivate()
            self.nav_mode.activate()
        else:
            self.rng_mode.activate()
            self.nav_mode.deactivate()
