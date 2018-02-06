from __future__ import absolute_import, division, print_function

import numpy as np

from glue.viewers.common.qt.data_viewer_with_state import DataViewerWithState
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.common.viz_client import init_mpl, update_appearance_from_settings
from glue.external.echo import delay_callback
from glue.utils import defer_draw, mpl_to_datetime64
from glue.utils.decorators import avoid_circular
from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.viewers.matplotlib.state import MatplotlibDataViewerState
from glue.viewers.image.layer_artist import ImageSubsetLayerArtist
from glue.core.edit_subset_mode import EditSubsetMode
from glue.core.command import ApplySubsetState

__all__ = ['MatplotlibDataViewer']


_MPL_LEFT_CLICK = 1
_MPL_RIGHT_CLICK = 3


# Eventually this should be defined elsewhere
class RoiSelectionMixin:

    def __init__(self):
        self._dc = None
        self._canvas = None
        self._edit_subset_mode = EditSubsetMode()

    def connect_mpl_events(self):
        self._canvas = self.figure.canvas
        self._dc = self.state.data_collection

        self._canvas.mpl_connect('button_press_event', self._button_press)
        self._canvas.mpl_connect('button_release_event', self._button_release)

    def _button_press(self, event):
        # Ignore button presses outside the data viewer canvas
        if event.xdata is None or event.ydata is None:
            return

        x, y = (int(event.xdata + 0.5), int(event.ydata + 0.5))

        roi_index = 0
        for layer in self.layers:
            if not isinstance(layer, ImageSubsetLayerArtist):
                continue
            roi = layer.state.layer.subset_state.roi
            if roi.contains(x, y):
                if event.button == _MPL_LEFT_CLICK:
                    print("HEY THERE", type(roi), hex(id(roi)))
                    self._select_roi(roi_index)
            roi_index += 1

    def _button_release(self, event):
        pass

    def _select_roi(self, index):
        self._edit_subset_mode.edit_subset = [self._dc.subset_groups[index]]


class MatplotlibDataViewer(DataViewerWithState, RoiSelectionMixin):

    _toolbar_cls = MatplotlibViewerToolbar
    _state_cls = MatplotlibDataViewerState

    def __init__(self, session, parent=None, wcs=None, state=None):

        super(MatplotlibDataViewer, self).__init__(session, parent, state=state)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self._axes = init_mpl(self.mpl_widget.canvas.fig, wcs=wcs)

        self.state.add_callback('aspect', self.update_aspect)

        self.update_aspect()

        self.state.add_callback('x_min', self.limits_to_mpl)
        self.state.add_callback('x_max', self.limits_to_mpl)
        self.state.add_callback('y_min', self.limits_to_mpl)
        self.state.add_callback('y_max', self.limits_to_mpl)

        self.limits_to_mpl()

        self.state.add_callback('x_log', self.update_x_log, priority=1000)
        self.state.add_callback('y_log', self.update_y_log, priority=1000)

        self.update_x_log()

        self.axes.callbacks.connect('xlim_changed', self.limits_from_mpl)
        self.axes.callbacks.connect('ylim_changed', self.limits_from_mpl)

        self.axes.set_autoscale_on(False)

        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

        self.connect_mpl_events()

    def redraw(self):
        self.figure.canvas.draw()

    @defer_draw
    def update_x_log(self, *args):
        self.axes.set_xscale('log' if self.state.x_log else 'linear')
        self.redraw()

    @defer_draw
    def update_y_log(self, *args):
        self.axes.set_yscale('log' if self.state.y_log else 'linear')
        self.redraw()

    def update_aspect(self, aspect=None):
        self.axes.set_aspect(self.state.aspect, adjustable='datalim')

    @avoid_circular
    def limits_from_mpl(self, *args):

        with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):

            if isinstance(self.state.x_min, np.datetime64):
                x_min, x_max = [mpl_to_datetime64(x) for x in self.axes.get_xlim()]
            else:
                x_min, x_max = self.axes.get_xlim()

            self.state.x_min, self.state.x_max = x_min, x_max

            if isinstance(self.state.y_min, np.datetime64):
                y_min, y_max = [mpl_to_datetime64(y) for y in self.axes.get_ylim()]
            else:
                y_min, y_max = self.axes.get_ylim()

            self.state.y_min, self.state.y_max = y_min, y_max

    @avoid_circular
    def limits_to_mpl(self, *args):
        if self.state.x_min is not None and self.state.x_max is not None:
            self.axes.set_xlim(self.state.x_min, self.state.x_max)
        if self.state.y_min is not None and self.state.y_max is not None:
            self.axes.set_ylim(self.state.y_min, self.state.y_max)

        if self.state.aspect == 'equal':

            # FIXME: for a reason I don't quite understand, dataLim doesn't
            # get updated immediately here, which means that there are then
            # issues in the first draw of the image (the limits are such that
            # only part of the image is shown). We just set dataLim manually
            # to avoid this issue.
            self.axes.dataLim.intervalx = self.axes.get_xlim()
            self.axes.dataLim.intervaly = self.axes.get_ylim()

            # We then force the aspect to be computed straight away
            self.axes.apply_aspect()

            # And propagate any changes back to the state since we have the
            # @avoid_circular decorator
            with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
                # TODO: fix case with datetime64 here
                self.state.x_min, self.state.x_max = self.axes.get_xlim()
                self.state.y_min, self.state.y_max = self.axes.get_ylim()

        self.axes.figure.canvas.draw()

    # TODO: shouldn't need this!
    @property
    def axes(self):
        return self._axes

    def _update_appearance_from_settings(self, message=None):
        update_appearance_from_settings(self.axes)
        self.redraw()

    def get_layer_artist(self, cls, layer=None, layer_state=None):
        return cls(self.axes, self.state, layer=layer, layer_state=layer_state)

    def apply_roi(self, roi):
        if len(self.layers) > 0:
            subset_state = self._roi_to_subset_state(roi)
            cmd = ApplySubsetState(data_collection=self._data,
                                   subset_state=subset_state)
            self._session.command_stack.do(cmd)
        else:
            # Make sure we force a redraw to get rid of the ROI
            self.axes.figure.canvas.draw()
