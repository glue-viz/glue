from glue.utils import defer_draw, decorate_all_methods, mpl_to_datetime64
from glue.viewers.matplotlib.qt.data_viewer import MatplotlibDataViewer
from glue.viewers.scatter.qt.layer_style_editor import ScatterLayerStyleEditor
from glue.viewers.scatter.layer_artist import ScatterLayerArtist
from glue.viewers.scatter.qt.options_widget import ScatterOptionsWidget
from glue.viewers.scatter.state import ScatterViewerState

from glue.viewers.scatter.viewer import MatplotlibScatterMixin

from echo import delay_callback


from numpy import rad2deg, deg2rad, datetime64

__all__ = ['ScatterViewer']


@decorate_all_methods(defer_draw)
class ScatterViewer(MatplotlibScatterMixin, MatplotlibDataViewer):

    LABEL = '2D Scatter'
    _layer_style_widget_cls = ScatterLayerStyleEditor
    _state_cls = ScatterViewerState
    _options_cls = ScatterOptionsWidget
    _data_artist_cls = ScatterLayerArtist
    _subset_artist_cls = ScatterLayerArtist

    tools = ['select:rectangle', 'select:xrange',
             'select:yrange', 'select:circle',
             'select:polygon']

    def __init__(self, session, parent=None, state=None):
        MatplotlibDataViewer.__init__(self, session, parent=parent, state=state)
        MatplotlibScatterMixin.setup_callbacks(self)

    def limits_to_mpl(self, *args):
        if self.state.plot_mode == 'rectilinear':
            super().limits_to_mpl(*args)
        elif self.state.plot_mode == 'polar':
            x_min, x_max = self.state.x_min, self.state.x_max

            y_min, y_max = self.state.y_min, self.state.y_max
            if self.state.y_log:
                if self.state.y_max <= 0:
                    y_min, y_max = 0.1, 1
                elif self.state.y_min <= 0:
                    y_min = y_max / 10

            self._skip_limits_from_mpl = True
            try:
                self.axes.set_thetalim(thetamin=rad2deg(x_min), thetamax=rad2deg(x_max))
                self.axes.set_ylim(y_min, y_max)
                self.axes.set_rorigin(y_min)
                self.axes.figure.canvas.draw_idle()
            finally:
                self._skip_limits_from_mpl = False
        else:
            pass

    def mpl_to_limits(self, *args):
        if self.state.plot_mode == 'rectilinear':
            super().mpl_to_limits(*args)
        elif self.state.plot_mode == 'polar':
            if self._skip_limits_from_mpl:
                return
            if isinstance(self.state.x_min, datetime64):
                x_min = mpl_to_datetime64(self.axes.get_theatmin())
                x_max = mpl_to_datetime64(self.axes.get_theatmax())
            else:
                x_min = self.axes.get_theatmin()
                x_max = self.axes.get_theatmax()

            if isinstance(self.state.y_min, datetime64):
                y_min, y_max = [mpl_to_datetime64(y) for y in self.axes.get_ylim()]
            else:
                y_min, y_max = self.axes.get_ylim()

            with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):

                self.state.x_min = x_min
                self.state.x_max = x_max
                self.state.y_min = y_min
                self.state.y_max = y_max

        else:
            pass

