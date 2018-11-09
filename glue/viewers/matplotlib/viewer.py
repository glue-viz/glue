from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib.patches import Rectangle

from glue.viewers.matplotlib.mpl_axes import update_appearance_from_settings
from glue.external.echo import delay_callback
from glue.utils import mpl_to_datetime64

__all__ = ['MatplotlibViewerMixin']

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


class MatplotlibViewerMixin(object):

    def setup_callbacks(self):

        for spine in self.axes.spines.values():
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

        self.state.add_callback('x_min', self._on_state_xlim_changed)
        self.state.add_callback('x_max', self._on_state_xlim_changed)
        self.state.add_callback('y_min', self._on_state_ylim_changed)
        self.state.add_callback('y_max', self._on_state_ylim_changed)

        self.state.add_callback('aspect', self._on_aspect_changed)

        self._on_aspect_changed()

        if (self.state.x_min or self.state.x_max or self.state.y_min or self.state.y_max) is None:
            self.limits_from_mpl()
        else:
            self.limits_to_mpl()

        self.state.add_callback('x_log', self.update_x_log, priority=1000)
        self.state.add_callback('y_log', self.update_y_log, priority=1000)

        self.update_x_log()

        self.axes.callbacks.connect('xlim_changed', self._on_mpl_xlim_changed)
        self.axes.callbacks.connect('ylim_changed', self._on_mpl_ylim_changed)
        self.figure.canvas.mpl_connect('resize_event', self._on_aspect_changed)

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

    def _update_computation(self, message=None):
        pass

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

    def _on_mpl_xlim_changed(self, *args):
        # If the x limits have been set, if the aspect ratio is 'equal', we
        # adjust the y limits to preserve the aspect ratio.
        self.limits_from_mpl(aspect_adjustable='y')

    def _on_mpl_ylim_changed(self, *args):
        # If the y limits have been set, if the aspect ratio is 'equal', we
        # adjust the x limits to preserve the aspect ratio.
        self.limits_from_mpl(aspect_adjustable='x')

    def limits_from_mpl(self, aspect_adjustable='x'):

        if getattr(self, '_skip_from_mpl', False):
            return

        if isinstance(self.state.x_min, np.datetime64):
            x_min, x_max = [mpl_to_datetime64(x) for x in self.axes.get_xlim()]
        else:
            x_min, x_max = self.axes.get_xlim()

        if isinstance(self.state.y_min, np.datetime64):
            y_min, y_max = [mpl_to_datetime64(y) for y in self.axes.get_ylim()]
        else:
            y_min, y_max = self.axes.get_ylim()

        x_min, x_max, y_min, y_max, changed = self.limits_with_aspect(x_min, x_max,
                                                                      y_min, y_max,
                                                                      aspect_adjustable=aspect_adjustable)

        # If the aspect ratio has caused some of the limits to change, we simply
        # change the relevant matplotlib limits and return - changing the
        # matplotlib limits will cause this method to be called again, at which
        # point the limits should no longer change, and will be set on the
        # state.
        if changed:
            if changed == 'x':
                self.axes.set_xlim(x_min, x_max)
            else:
                self.axes.set_ylim(y_min, y_max)
            self.axes.figure.canvas.draw()
            return

        with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.state.x_min = x_min
            self.state.x_max = x_max
            self.state.y_min = y_min
            self.state.y_max = y_max

    def _on_state_xlim_changed(self, *args):
        self.limits_to_mpl(aspect_adjustable='y')

    def _on_state_ylim_changed(self, *args):
        self.limits_to_mpl(aspect_adjustable='x')

    def limits_to_mpl(self, aspect_adjustable='x'):

        if (self.state.x_min is None or self.state.x_max is None or
            self.state.y_min is None or self.state.y_max is None):
            return

        x_min, x_max = self.state.x_min, self.state.x_max
        if self.state.x_log:
            if self.state.x_max <= 0:
                x_min, x_max = 0.1, 1
            elif self.state.x_min <= 0:
                x_min = x_max / 10

        y_min, y_max = self.state.y_min, self.state.y_max
        if self.state.y_log:
            if self.state.y_max <= 0:
                y_min, y_max = 0.1, 1
            elif self.state.y_min <= 0:
                y_min = y_max / 10

        x_min, x_max, y_min, y_max, changed = self.limits_with_aspect(x_min, x_max,
                                                                      y_min, y_max,
                                                                      aspect_adjustable=aspect_adjustable)

        # If the limits have changed, we just change them back on the state
        if changed:
            with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
                self.state.x_min = x_min
                self.state.x_max = x_max
                self.state.y_min = y_min
                self.state.y_max = y_max
            return

        self._skip_from_mpl = True
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)
        self._skip_from_mpl = False

        self.axes.figure.canvas.draw()

    @property
    def axes_ratio(self):

        # Get figure aspect ratio
        width, height = self.figure.get_size_inches()
        fig_aspect = height / width

        # Get axes aspect ratio
        l, b, w, h = self.axes.get_position().bounds
        axes_ratio = fig_aspect * (h / w)

        return axes_ratio

    def limits_with_aspect(self, x_min, x_max, y_min, y_max, aspect_adjustable='auto'):
        """
        Determine whether the limits need to be changed based on the aspect ratio.
        """

        changed = None

        if self.state.aspect == 'equal':

            # Find axes aspect ratio
            axes_ratio = self.axes_ratio

            # Find current data ratio
            data_ratio = abs(y_max - y_min) / abs(x_max - x_min)

            # Only do something if the data ratio is sufficiently different
            # from the axes ratio.
            if abs(data_ratio - axes_ratio) / (0.5 * (data_ratio + axes_ratio)) > 0.01:

                # We now adjust the limits - which ones we adjust depends on
                # the adjust keyword. We also make sure we preserve the
                # mid-point of the current coordinates.

                if aspect_adjustable == 'both':

                    # We need to adjust both at the same time

                    x_mid = 0.5 * (x_min + x_max)
                    x_width = abs(x_max - x_min) * (data_ratio / axes_ratio) ** 0.5

                    y_mid = 0.5 * (y_min + y_max)
                    y_width = abs(y_max - y_min) / (data_ratio / axes_ratio) ** 0.5

                    x_min = x_mid - x_width / 2.
                    x_max = x_mid + x_width / 2.

                    y_min = y_mid - y_width / 2.
                    y_max = y_mid + y_width / 2.

                    changed = 'both'

                elif (aspect_adjustable == 'auto' and data_ratio > axes_ratio) or aspect_adjustable == 'x':
                    x_mid = 0.5 * (x_min + x_max)
                    x_width = abs(y_max - y_min) / axes_ratio
                    x_min = x_mid - x_width / 2.
                    x_max = x_mid + x_width / 2.
                    changed = 'x'
                else:
                    y_mid = 0.5 * (y_min + y_max)
                    y_width = abs(x_max - x_min) * axes_ratio
                    y_min = y_mid - y_width / 2.
                    y_max = y_mid + y_width / 2.
                    changed = 'y'

        data_ratio = abs(y_max - y_min) / abs(x_max - x_min)

        return x_min, x_max, y_min, y_max, changed

    def _on_aspect_changed(self, *args):

        if (self.state.x_min is None or self.state.x_max is None or
            self.state.y_min is None or self.state.y_max is None):
            return

        x_min, x_max, y_min, y_max, changed = self.limits_with_aspect(self.state.x_min,
                                                                      self.state.x_max,
                                                                      self.state.y_min,
                                                                      self.state.y_max,
                                                                      aspect_adjustable='both')

        if changed:
            with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):
                self.state.x_min = x_min
                self.state.x_max = x_max
                self.state.y_min = y_min
                self.state.y_max = y_max


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
