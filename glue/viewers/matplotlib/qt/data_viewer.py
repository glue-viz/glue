from __future__ import absolute_import, division, print_function

from glue.viewers.common.qt.data_viewer_with_state import DataViewerWithState
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.common.viz_client import init_mpl, update_appearance_from_settings
from glue.external.echo import delay_callback
from glue.utils import nonpartial
from glue.utils.decorators import avoid_circular
from glue.viewers.matplotlib.qt.toolbar import MatplotlibViewerToolbar
from glue.viewers.matplotlib.state import MatplotlibDataViewerState

__all__ = ['MatplotlibDataViewer']


class MatplotlibDataViewer(DataViewerWithState):

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

        self.state.add_callback('x_min', nonpartial(self.limits_to_mpl))
        self.state.add_callback('x_max', nonpartial(self.limits_to_mpl))
        self.state.add_callback('y_min', nonpartial(self.limits_to_mpl))
        self.state.add_callback('y_max', nonpartial(self.limits_to_mpl))

        self.limits_to_mpl()

        self.state.add_callback('x_log', nonpartial(self.update_x_log))
        self.state.add_callback('y_log', nonpartial(self.update_y_log))

        self.update_x_log()

        self.axes.callbacks.connect('xlim_changed', nonpartial(self.limits_from_mpl))
        self.axes.callbacks.connect('ylim_changed', nonpartial(self.limits_from_mpl))

        self.axes.set_autoscale_on(False)

        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

    def redraw(self):
        self.figure.canvas.draw()

    def update_x_log(self):
        self.axes.set_xscale('log' if self.state.x_log else 'linear')

    def update_y_log(self):
        self.axes.set_yscale('log' if self.state.y_log else 'linear')

    def update_aspect(self, aspect=None):
        self.axes.set_aspect(self.state.aspect, adjustable='datalim')

    @avoid_circular
    def limits_from_mpl(self):
        with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.state.x_min, self.state.x_max = self.axes.get_xlim()
            self.state.y_min, self.state.y_max = self.axes.get_ylim()

    @avoid_circular
    def limits_to_mpl(self):
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
