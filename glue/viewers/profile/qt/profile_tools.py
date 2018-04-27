from __future__ import absolute_import, division, print_function

import os
import weakref
import traceback
from collections import OrderedDict

import numpy as np

from qtpy.QtCore import Qt
from qtpy import QtWidgets, QtGui

from glue.utils import color2hex, nanmean, nanmedian, nanmin, nanmax, nansum
from glue.config import fit_plugin, viewer_tool
from glue.utils.qt import load_ui, fix_tab_widget_fontsize
from glue.viewers.profile.qt.mouse_mode import NavigateMouseMode, RangeMouseMode
from glue.core.qt.fitters import FitSettingsWidget
from glue.utils.qt import Worker
from glue.viewers.common.qt.tool import Tool
from glue.viewers.image.state import AggregateSlice
from glue.core.aggregate import mom1, mom2
from glue.core import Data, Subset
from glue.viewers.image.qt import ImageViewer
from glue.core.link_manager import is_convertible_to_single_pixel_cid
from glue.external.echo import SelectionCallbackProperty
from glue.external.echo.qt import connect_combo_selection

__all__ = ['ProfileTools']


MODES = ['navigate', 'fit', 'collapse']

COLLAPSE_FUNCS = OrderedDict([(nanmean, 'Mean'),
                              (nanmedian, 'Median'),
                              (nanmin, 'Minimum'),
                              (nanmax, 'Maximum'),
                              (nansum, 'Sum'),
                              (mom1, 'Moment 1'),
                              (mom2, 'Moment 2')])


@viewer_tool
class ProfileAnalysisTool(Tool):

    action_text = 'Options'
    tool_id = 'profile-analysis'

    def __init__(self, viewer):
        super(ProfileAnalysisTool, self).__init__(viewer)
        self._profile_tools = ProfileTools(viewer)
        container_widget = QtWidgets.QSplitter(Qt.Horizontal)
        plot_widget = viewer.centralWidget()
        container_widget.addWidget(plot_widget)
        container_widget.addWidget(self._profile_tools)
        viewer.setCentralWidget(container_widget)
        self._profile_tools.enable()
        self._profile_tools.hide()

    def activate(self):
        if self._profile_tools.isVisible():
            self._profile_tools.hide()
        else:
            self._profile_tools.show()


class ProfileTools(QtWidgets.QWidget):

    fit_function = SelectionCallbackProperty()
    collapse_function = SelectionCallbackProperty()

    def __init__(self, parent=None):

        super(ProfileTools, self).__init__(parent=parent)

        self.ui = load_ui('profile_tools.ui', self,
                          directory=os.path.dirname(__file__))

        fix_tab_widget_fontsize(self.ui.tabs)

        self._viewer = weakref.ref(parent)
        self.image_viewer = None

    @property
    def viewer(self):
        return self._viewer()

    def show(self, *args):
        super(ProfileTools, self).show(*args)
        self._on_tab_change()

    def hide(self, *args):
        super(ProfileTools, self).hide(*args)
        self.rng_mode.deactivate()
        self.nav_mode.deactivate()

    def enable(self):

        self.nav_mode = NavigateMouseMode(self.viewer,
                                          press_callback=self._on_nav_activate)
        self.rng_mode = RangeMouseMode(self.viewer)

        self.nav_mode.state.add_callback('x', self._on_slider_change)

        self.ui.tabs.setCurrentIndex(0)

        self.ui.tabs.currentChanged.connect(self._on_tab_change)

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

        ProfileTools.fit_function.set_choices(self, list(fit_plugin))
        ProfileTools.fit_function.set_display_func(self, lambda fitter: fitter.label)
        connect_combo_selection(self, 'fit_function', self.ui.combosel_fit_function)

        ProfileTools.collapse_function.set_choices(self, list(COLLAPSE_FUNCS))
        ProfileTools.collapse_function.set_display_func(self, COLLAPSE_FUNCS.get)
        connect_combo_selection(self, 'collapse_function', self.ui.combosel_collapse_function)

        self._toolbar_connected = False

        self.viewer.toolbar_added.connect(self._on_toolbar_added)

        self.viewer.state.add_callback('x_att', self._on_x_att_change)

    def _on_x_att_change(self, *event):
        self.nav_mode.clear()
        self.rng_mode.clear()

    def _on_nav_activate(self, *args):
        self._nav_data = self._visible_data()
        self._nav_viewers = {}
        for data in self._nav_data:
            pix_cid = is_convertible_to_single_pixel_cid(data, self.viewer.state.x_att)
            self._nav_viewers[data] = self._viewers_with_data_slice(data, pix_cid)

    def _on_slider_change(self, *args):

        x = self.nav_mode.state.x

        if x is None:
            return

        for data in self._nav_data:

            axis, slc = self._get_axis_and_pixel_slice(data, x)

            for viewer in self._nav_viewers[data]:
                slices = list(viewer.state.slices)
                slices[axis] = slc
                viewer.state.slices = tuple(slices)

    def _get_axis_and_pixel_slice(self, data, x):

        if self.viewer.state.x_att in data.pixel_component_ids:
            axis = self.viewer.state.x_att.axis
            slc = int(round(x))
        else:
            pix_cid = is_convertible_to_single_pixel_cid(data, self.viewer.state.x_att)
            axis = pix_cid.axis
            axis_view = [0] * data.ndim
            axis_view[pix_cid.axis] = slice(None)
            axis_values = data[self.viewer.state.x_att, axis_view]
            slc = int(np.argmin(np.abs(axis_values - x)))

        return axis, slc

    def _on_settings(self):
        d = FitSettingsWidget(self.fit_function())
        d.exec_()

    def _on_fit(self):
        """
        Fit a model to the data

        The fitting happens on a dedicated thread, to keep the UI
        responsive
        """

        if self.rng_mode.state.x_min is None or self.rng_mode.state.x_max is None:
            return

        x_range = self.rng_mode.state.x_range
        fitter = self.fit_function()

        def on_success(result):
            fit_results, x, y = result
            report = ""
            normalize = {}
            for layer_artist in fit_results:
                report += ("<b><font color='{0}'>{1}</font>"
                           "</b>".format(color2hex(layer_artist.state.color),
                                         layer_artist.layer.label))
                report += "<pre>" + fitter.summarize(fit_results[layer_artist], x, y) + "</pre>"
                if self.viewer.state.normalize:
                    normalize[layer_artist] = layer_artist.state.normalize_values
            self._report_fit(report)
            self._plot_fit(fitter, fit_results, x, y, normalize)

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

    def wait_for_fit(self):
        self._fit_worker.wait()

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

    def _plot_fit(self, fitter, fit_result, x, y, normalize):

        self._clear_fit()

        for layer in fit_result:
            # y_model = fitter.predict(fit_result[layer], x)
            self._fit_artists.append(fitter.plot(fit_result[layer], self.axes, x,
                                                 alpha=layer.state.alpha,
                                                 linewidth=layer.state.linewidth * 0.5,
                                                 color=layer.state.color,
                                                 normalize=normalize.get(layer, None))[0])

        self.canvas.draw()

    def _visible_data(self):
        datasets = set()
        for layer_artist in self.viewer.layers:
            if layer_artist.enabled and layer_artist.visible:
                if isinstance(layer_artist.state.layer, Data):
                    datasets.add(layer_artist.state.layer)
                elif isinstance(layer_artist.state.layer, Subset):
                    datasets.add(layer_artist.state.layer.data)
        return list(datasets)

    def _viewers_with_data_slice(self, data, xatt):

        if self.viewer.session.application is None:
            return []

        viewers = []
        for tab in self.viewer.session.application.viewers:
            for viewer in tab:
                if isinstance(viewer, ImageViewer):
                    for layer_artist in viewer._layer_artist_container[data]:
                        if layer_artist.enabled and layer_artist.visible:
                            if len(viewer.state.slices) >= xatt.axis:
                                viewers.append(viewer)
        return viewers

    def _on_collapse(self):

        if self.rng_mode.state.x_min is None or self.rng_mode.state.x_max is None:
            return

        func = self.collapse_function
        x_range = self.rng_mode.state.x_range

        for data in self._visible_data():

            pix_cid = is_convertible_to_single_pixel_cid(data, self.viewer.state.x_att)

            for viewer in self._viewers_with_data_slice(data, pix_cid):

                slices = list(viewer.state.slices)

                # TODO: don't need to fetch axis twice
                axis, imin = self._get_axis_and_pixel_slice(data, x_range[0])
                axis, imax = self._get_axis_and_pixel_slice(data, x_range[1])

                current_slice = slices[axis]

                if isinstance(current_slice, AggregateSlice):
                    current_slice = current_slice.center

                slices[axis] = AggregateSlice(slice(imin, imax),
                                              current_slice,
                                              func)

                viewer.state.slices = tuple(slices)

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
