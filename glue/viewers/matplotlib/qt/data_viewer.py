from __future__ import absolute_import, division, print_function

import numpy as np
from qtpy.QtCore import QTimer

from matplotlib.patches import Rectangle

from glue.core.message import ComputationStartedMessage
from glue.viewers.common.qt.data_viewer import DataViewer
from glue.viewers.matplotlib.qt.widget import MplWidget
from glue.viewers.matplotlib.mpl_axes import init_mpl, update_appearance_from_settings
from glue.external.echo import delay_callback
from glue.utils import defer_draw, mpl_to_datetime64, avoid_circular, decorate_all_methods
from glue.viewers.matplotlib.state import MatplotlibDataViewerState

# The following import is required to register the viewer tools
from glue.viewers.matplotlib.qt import toolbar  # noqa

__all__ = ['MatplotlibDataViewer']

SCRIPT_HEADER = """
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='{aspect}')
""".strip()

SCRIPT_FOOTER = """
# Set limits
ax.set_xlim({x_min}, {x_max})
ax.set_ylim({y_min}, {y_max})

# Set scale (log or linear)
ax.set_xscale('{x_log_str}')
ax.set_yscale('{y_log_str}')

# Set axis label properties
ax.set_xlabel('{x_axislabel}', weight='{x_axislabel_weight}', size={x_axislabel_size})
ax.set_ylabel('{y_axislabel}', weight='{y_axislabel_weight}', size={y_axislabel_size})

# Set tick label properties
ax.tick_params('x', labelsize={x_ticklabel_size})
ax.tick_params('y', labelsize={x_ticklabel_size})

# Save figure
fig.savefig('glue_plot.png')
plt.close(fig)
""".strip()

ZORDER_MAX = 100000


@decorate_all_methods(defer_draw)
class MatplotlibDataViewer(DataViewer):

    _state_cls = MatplotlibDataViewerState

    tools = ['mpl:home', 'mpl:pan', 'mpl:zoom']
    subtools = {'save': ['mpl:save']}

    def __init__(self, session, parent=None, wcs=None, state=None):

        super(MatplotlibDataViewer, self).__init__(session, parent=parent, state=state)

        # Use MplWidget to set up a Matplotlib canvas inside the Qt window
        self.mpl_widget = MplWidget()
        self.setCentralWidget(self.mpl_widget)

        # TODO: shouldn't have to do this
        self.central_widget = self.mpl_widget

        self.figure, self._axes = init_mpl(self.mpl_widget.canvas.fig, wcs=wcs)

        for spine in self._axes.spines.values():
            spine.set_zorder(ZORDER_MAX)

        self.loading_rectangle = Rectangle((0, 0), 1, 1, color='0.9', alpha=0.9,
                                           zorder=ZORDER_MAX - 1, transform=self.axes.transAxes)
        self.loading_rectangle.set_visible(False)
        self.axes.add_patch(self.loading_rectangle)

        self.loading_text = self.axes.text(0.4, 0.5, 'Computing', color='k',
                                           zorder=self.loading_rectangle.get_zorder() + 1,
                                           ha='left', va='center',
                                           transform=self.axes.transAxes)
        self.loading_text.set_visible(False)

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

        self.state.add_callback('x_axislabel', self.update_x_axislabel)
        self.state.add_callback('x_axislabel_weight', self.update_x_axislabel)
        self.state.add_callback('x_axislabel_size', self.update_x_axislabel)

        self.state.add_callback('y_axislabel', self.update_y_axislabel)
        self.state.add_callback('y_axislabel_weight', self.update_y_axislabel)
        self.state.add_callback('y_axislabel_size', self.update_y_axislabel)

        self.state.add_callback('x_ticklabel_size', self.update_x_ticklabel)
        self.state.add_callback('y_ticklabel_size', self.update_y_ticklabel)

        self.update_x_axislabel()
        self.update_y_axislabel()
        self.update_x_ticklabel()
        self.update_y_ticklabel()

        self.central_widget.resize(600, 400)
        self.resize(self.central_widget.size())

        self._monitor_computation = QTimer()
        self._monitor_computation.setInterval(500)
        self._monitor_computation.timeout.connect(self._update_computation)

    def _update_computation(self, message=None):

        # If we get a ComputationStartedMessage and the timer isn't currently
        # active, then we start the timer but we then return straight away.
        # This is to avoid showing the 'Computing' message straight away in the
        # case of reasonably fast operations.
        if isinstance(message, ComputationStartedMessage):
            if not self._monitor_computation.isActive():
                self._monitor_computation.start()
            return

        for layer_artist in self.layers:
            if layer_artist.is_computing:
                self.loading_rectangle.set_visible(True)
                text = self.loading_text.get_text()
                if text.count('.') > 2:
                    text = 'Computing'
                else:
                    text += '.'
                self.loading_text.set_text(text)
                self.loading_text.set_visible(True)
                self.redraw()
                return

        self.loading_rectangle.set_visible(False)
        self.loading_text.set_visible(False)
        self.redraw()

        # If we get here, the computation has stopped so we can stop the timer
        self._monitor_computation.stop()

    def add_data(self, *args, **kwargs):
        return super(MatplotlibDataViewer, self).add_data(*args, **kwargs)

    def add_subset(self, *args, **kwargs):
        return super(MatplotlibDataViewer, self).add_subset(*args, **kwargs)

    def update_x_axislabel(self, *event):
        self.axes.set_xlabel(self.state.x_axislabel,
                             weight=self.state.x_axislabel_weight,
                             size=self.state.x_axislabel_size)
        self.redraw()

    def update_y_axislabel(self, *event):
        self.axes.set_ylabel(self.state.y_axislabel,
                             weight=self.state.y_axislabel_weight,
                             size=self.state.y_axislabel_size)
        self.redraw()

    def update_x_ticklabel(self, *event):
        self.axes.tick_params(axis='x', labelsize=self.state.x_ticklabel_size)
        self.axes.xaxis.get_offset_text().set_fontsize(self.state.x_ticklabel_size)
        self.redraw()

    def update_y_ticklabel(self, *event):
        self.axes.tick_params(axis='y', labelsize=self.state.y_ticklabel_size)
        self.axes.yaxis.get_offset_text().set_fontsize(self.state.y_ticklabel_size)
        self.redraw()

    def redraw(self):
        self.figure.canvas.draw()

    def update_x_log(self, *args):
        self.axes.set_xscale('log' if self.state.x_log else 'linear')
        self.redraw()

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
            x_min, x_max = self.state.x_min, self.state.x_max
            if self.state.x_log:
                if self.state.x_max <= 0:
                    x_min, x_max = 0.1, 1
                elif self.state.x_min <= 0:
                    x_min = x_max / 10
            self.axes.set_xlim(x_min, x_max)

        if self.state.y_min is not None and self.state.y_max is not None:
            y_min, y_max = self.state.y_min, self.state.y_max
            if self.state.y_log:
                if self.state.y_max <= 0:
                    y_min, y_max = 0.1, 1
                elif self.state.y_min <= 0:
                    y_min = y_max / 10
            self.axes.set_ylim(y_min, y_max)

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

    def apply_roi(self, roi, override_mode=None):
        """ This method must be implemented by subclasses """
        raise NotImplementedError

    def _script_header(self):
        state_dict = self.state.as_dict()
        return ['import matplotlib.pyplot as plt'], SCRIPT_HEADER.format(**state_dict)

    def _script_footer(self):
        state_dict = self.state.as_dict()
        state_dict['x_log_str'] = 'log' if self.state.x_log else 'linear'
        state_dict['y_log_str'] = 'log' if self.state.y_log else 'linear'
        return [], SCRIPT_FOOTER.format(**state_dict)
