from textwrap import indent

import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.artist import setp as msetp

from glue.config import settings
from glue.viewers.matplotlib.mpl_axes import update_appearance_from_settings
from echo import delay_callback
from glue.utils import mpl_to_datetime64

__all__ = ['MatplotlibViewerMixin']

SCRIPT_HEADER = """
# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='{aspect}', projection={projection})

# for the legend
legend_handles = []
legend_labels = []
legend_handler_dict = dict()
""".strip()

LIMIT_SCRIPT = """
# Set limits
ax.set_xlim(left={x_min}, right={x_max})
ax.set_ylim(bottom={y_min}, top={y_max})
"""

SCRIPT_FOOTER = """
{limit_script}

{scale_script}

# Set axis label properties
ax.set_xlabel('{x_axislabel}', weight='{x_axislabel_weight}', size={x_axislabel_size})
ax.set_ylabel('{y_axislabel}', weight='{y_axislabel_weight}', size={y_axislabel_size})

# Set tick label properties
ax.tick_params('x', labelsize={x_ticklabel_size})
ax.tick_params('y', labelsize={y_ticklabel_size})

# Set the figure colors
# To create the figure with the default settings, comment out the next line
# Alternatively, to create the figure with colors from the current glue environment:
#  - from glue.config import settings
#  - set_figure_colors(ax, background=settings.BACKGROUND_COLOR, foreground=settings.FOREGROUND_COLOR
set_figure_colors(ax, background={background_color}, foreground={foreground_color})

# For manual edition of the plot
#  - Uncomment the next code line (plt.show)
#  - Also change the matplotlib backend to qt5Agg
#  - And comment the "plt.close" line
# plt.show()

# Save figure
fig.savefig('glue_plot.png')
plt.close(fig)
""".strip()

SCRIPT_LEGEND = """
ax.legend(legend_handles, legend_labels,
    handler_map=legend_handler_dict,
    loc='{location}',            # location
    framealpha={alpha:.2f},      # opacity of the frame
    title='{title}',             # title of the legend
    title_fontsize={fontsize},   # fontsize of the title
    fontsize={fontsize},          # fontsize of the labels
    facecolor='{frame_color}',
    labelcolor='{label_color}',
    edgecolor={edge_color}
)
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

        self.state.add_callback('x_min', self.limits_to_mpl)
        self.state.add_callback('x_max', self.limits_to_mpl)
        self.state.add_callback('y_min', self.limits_to_mpl)
        self.state.add_callback('y_max', self.limits_to_mpl)

        if (self.state.x_min or self.state.x_max or self.state.y_min or self.state.y_max) is None:
            self.limits_from_mpl()
        else:
            self.limits_to_mpl()

        self.state.add_callback('x_log', self.update_x_log, priority=1000)
        self.state.add_callback('y_log', self.update_y_log, priority=1000)

        self.update_x_log()
        self.update_y_log()

        self.axes.callbacks.connect('xlim_changed', self.limits_from_mpl)
        self.axes.callbacks.connect('ylim_changed', self.limits_from_mpl)
        self.figure.canvas.mpl_connect('resize_event', self._on_resize)

        self._on_resize()

        self.axes.set_autoscale_on(False)

        self.state.add_callback('x_axislabel', self.update_x_axislabel)
        self.state.add_callback('x_axislabel_weight', self.update_x_axislabel)
        self.state.add_callback('x_axislabel_size', self.update_x_axislabel)

        self.state.add_callback('y_axislabel', self.update_y_axislabel)
        self.state.add_callback('y_axislabel_weight', self.update_y_axislabel)
        self.state.add_callback('y_axislabel_size', self.update_y_axislabel)

        self.state.add_callback('x_ticklabel_size', self.update_x_ticklabel)
        self.state.add_callback('y_ticklabel_size', self.update_y_ticklabel)

        self.state.legend.add_callback('visible', self.draw_legend)
        self.state.legend.add_callback('location', self.draw_legend)
        self.state.legend.add_callback('alpha', self.update_legend)
        self.state.legend.add_callback('title', self.draw_legend)
        self.state.legend.add_callback('fontsize', self.draw_legend)
        self.state.legend.add_callback('frame_color', self.update_legend)
        self.state.legend.add_callback('show_edge', self.update_legend)
        self.state.legend.add_callback('text_color', self.update_legend)

        self.update_x_axislabel()
        self.update_y_axislabel()
        self.update_x_ticklabel()
        self.update_y_ticklabel()
        self.draw_legend()

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

    def get_handles_legend(self):
        """Collect the handles and labels from each layer artist."""
        handles = []
        labels = []
        handler_dict = {}
        for layer_artist in self._layer_artist_container:
            handle, label, handler = layer_artist.get_handle_legend()
            if handle is not None:
                handles.append(handle)
                labels.append(label)
                if handler is not None:
                    handler_dict[handle] = handler
        return handles, labels, handler_dict

    def _update_legend_visual(self, legend):
        """Update the legend colors and opacity. No redraw."""
        msetp(legend.get_title(), color=self.state.legend.text_color)
        msetp(legend.get_texts(), color=self.state.legend.text_color)
        msetp(legend.get_frame(),
              alpha=self.state.legend.alpha,
              facecolor=self.state.legend.frame_color,
              edgecolor=self.state.legend.edge_color
              )

    def update_legend(self, *args):
        """Update the legend colors and opacity."""
        legend = self.axes.get_legend()
        if legend is not None:
            self._update_legend_visual(legend)
        self.redraw()

    def draw_legend(self, *args):
        if self.state.legend.visible:
            handles, labels, handler_map = self.get_handles_legend()
            kwargs = dict(loc=self.state.legend.mpl_location,
                          title=self.state.legend.title,
                          title_fontsize=self.state.legend.fontsize,
                          fontsize=self.state.legend.fontsize)
            if handler_map is not None:
                kwargs["handler_map"] = handler_map
            legend = self.axes.legend(handles, labels, **kwargs)
            self._update_legend_visual(legend)
            legend.set_draggable(self.state.legend.draggable)
        else:
            legend = self.axes.get_legend()
            if legend is not None:
                legend.remove()
        self.redraw()

    def redraw(self):
        self.figure.canvas.draw_idle()

    def update_x_log(self, *args):
        self.axes.set_xscale('log' if self.state.x_log else 'linear')
        self.redraw()

    def update_y_log(self, *args):
        self.axes.set_yscale('log' if self.state.y_log else 'linear')
        self.redraw()

    def limits_from_mpl(self, *args, **kwargs):

        if getattr(self, '_skip_limits_from_mpl', False):
            return

        if isinstance(self.state.x_min, np.datetime64):
            x_min, x_max = [mpl_to_datetime64(x) for x in self.axes.get_xlim()]
        else:
            x_min, x_max = self.axes.get_xlim()

        if isinstance(self.state.y_min, np.datetime64):
            y_min, y_max = [mpl_to_datetime64(y) for y in self.axes.get_ylim()]
        else:
            y_min, y_max = self.axes.get_ylim()

        with delay_callback(self.state, 'x_min', 'x_max', 'y_min', 'y_max'):

            self.state.x_min = x_min
            self.state.x_max = x_max
            self.state.y_min = y_min
            self.state.y_max = y_max

    def limits_to_mpl(self, *args):

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

        # Since we deal with aspect ratio internally, there are no conditions
        # under which we would want to immediately change the state once the
        # limits are set in Matplotlib, so we avoid this by setting the
        # _skip_limits_from_mpl attribute which is then recognized inside
        # limits_from_mpl
        self._skip_limits_from_mpl = True
        try:
            self.axes.set_xlim(x_min, x_max)
            self.axes.set_ylim(y_min, y_max)
            self.axes.figure.canvas.draw_idle()
        finally:
            self._skip_limits_from_mpl = False

    @property
    def axes_ratio(self):

        # Get figure aspect ratio
        width, height = self.figure.get_size_inches()
        fig_aspect = height / width

        # Get axes aspect ratio
        l, b, w, h = self.axes.get_position().bounds
        axes_ratio = fig_aspect * (h / w)

        return axes_ratio

    def _on_resize(self, *args):
        self.state._set_axes_aspect_ratio(self.axes_ratio)

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
        proj = "'" + self.state.plot_mode + "'" if hasattr(self.state, 'plot_mode') else "None"
        state_dict['projection'] = proj
        return ['import matplotlib',
                "matplotlib.use('Agg')",
                "# matplotlib.use('qt5Agg')",
                'import matplotlib.pyplot as plt',
                'from glue.viewers.matplotlib.mpl_axes import set_figure_colors'], SCRIPT_HEADER.format(**state_dict)

    def _script_footer(self):
        mode = 'rectilinear' if not hasattr(self.state, 'plot_mode') else self.state.plot_mode
        state_dict = self.state.as_dict()
        temp_str = ''
        if mode not in ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar']:
            x_log_str = 'log' if self.state.x_log else 'linear'
            temp_str += "ax.set_xscale('" + x_log_str + "')\n"
        if mode not in ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar']:
            y_log_str = 'log' if self.state.y_log else 'linear'
            temp_str += "ax.set_yscale('" + y_log_str + "')\n"
        if mode == 'polar':
            state_dict['x_axislabel'] = ''
            state_dict['y_axislabel'] = ''
        state_dict['background_color'] = "\"{color}\"".format(color=settings.BACKGROUND_COLOR)
        state_dict['foreground_color'] = "\"{color}\"".format(color=settings.FOREGROUND_COLOR)
        state_dict['scale_script'] = "# Set scale (log or linear)\n" + temp_str if temp_str else ''
        full_sphere = ['aitoff', 'hammer', 'lambert', 'mollweide']
        state_dict['limit_script'] = '' if mode in full_sphere else LIMIT_SCRIPT.format(**state_dict)
        return [], SCRIPT_FOOTER.format(**state_dict)

    def _script_legend(self):
        state_dict = self.state.legend.as_dict()
        state_dict['location'] = self.state.legend.mpl_location
        state_dict['edge_color'] = self.state.legend.edge_color
        state_dict['label_color'] = settings.FOREGROUND_COLOR
        legend_str = SCRIPT_LEGEND.format(**state_dict)
        if not self.state.legend.visible:
            legend_str = indent(legend_str, "# ")
        return [], legend_str
