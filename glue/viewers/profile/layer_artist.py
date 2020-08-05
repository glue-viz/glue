import sys
import warnings

import numpy as np
from matplotlib.lines import Line2D

from glue.core import BaseData
from glue.utils import defer_draw, nanmin, nanmax
from glue.viewers.profile.state import ProfileLayerState
from glue.viewers.matplotlib.layer_artist import MatplotlibLayerArtist
from glue.core.exceptions import IncompatibleAttribute, IncompatibleDataException
from glue.viewers.profile.python_export import python_export_profile_layer


class ProfileLayerArtist(MatplotlibLayerArtist):

    _layer_state_cls = ProfileLayerState
    _python_exporter = python_export_profile_layer

    def __init__(self, axes, viewer_state, layer_state=None, layer=None):

        super(ProfileLayerArtist, self).__init__(axes, viewer_state,
                                                 layer_state=layer_state, layer=layer)

        # Watch for changes in the viewer state which would require the
        # layers to be redrawn
        self._viewer_state.add_global_callback(self._update_profile)
        self.state.add_global_callback(self._update_profile)

        self.plot_artist = self.axes.plot([1, 2, 3], [3, 4, 5], 'k-', drawstyle='steps-mid',
                                          color=self.state.layer.style.color)[0]
        self.mpl_artists = [self.plot_artist]

    @defer_draw
    def _calculate_profile(self, reset=False):
        try:
            self.notify_start_computation()
            self._calculate_profile_thread(reset=reset)
        except Exception:
            self._calculate_profile_error(sys.exc_info())
        else:
            self._calculate_profile_postthread()

    def _calculate_profile_thread(self, reset=False):
        # We need to ignore any warnings that happen inside the thread
        # otherwise the thread tries to send these to the glue logger (which
        # uses Qt), which then results in this kind of error:
        # QObject::connect: Cannot queue arguments of type 'QTextCursor'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if reset:
                self.state.reset_cache()
            self.state.update_profile(update_limits=False)

    def _calculate_profile_postthread(self):

        self.notify_end_computation()

        # It's possible for this method to get called but for the state to have
        # been updated in the mean time to have a histogram that raises an
        # exception (for example an IncompatibleAttribute). If any errors happen
        # here, we simply ignore them since _calculate_histogram_error will get
        # called directly.
        try:
            visible_data = self.state.profile
        except Exception:
            return

        self.enable()

        # The following can happen if self.state.visible is None - in this case
        # we just terminate early. If the visible property is changed, it will
        # trigger the _calculate_profile code to re-run.
        if visible_data is None:
            return

        x, y = visible_data

        # Update the data values.
        if len(x) > 0:
            x = np.arange(len(x))
            self.state.update_limits()
            # Normalize profile values to the [0:1] range based on limits
            if self._viewer_state.normalize:
                y = self.state.normalize_values(y)
            self.plot_artist.set_data(x, y)
        else:
            # We need to do this otherwise we get issues on Windows when
            # passing an empty list to plot_artist
            self.plot_artist.set_data([0.], [0.])

        # TODO: the following was copy/pasted from the histogram viewer, maybe
        # we can find a way to avoid duplication?

        # We have to do the following to make sure that we reset the y_max as
        # needed. We can't simply reset based on the maximum for this layer
        # because other layers might have other values, and we also can't do:
        #
        #   self._viewer_state.y_max = max(self._viewer_state.y_max, result[0].max())
        #
        # because this would never allow y_max to get smaller.

        if not self._viewer_state.normalize and len(y) > 0:

            y_min = nanmin(y)
            y_max = nanmax(y)
            y_range = y_max - y_min

            self.state._y_min = y_min - y_range * 0.1
            self.state._y_max = y_max + y_range * 0.1

            largest_y_max = max(getattr(layer, '_y_max', 0) for layer in self._viewer_state.layers)
            if largest_y_max != self._viewer_state.y_max:
                self._viewer_state.y_max = largest_y_max

            smallest_y_min = min(getattr(layer, '_y_min', np.inf) for layer in self._viewer_state.layers)
            if smallest_y_min != self._viewer_state.y_min:
                self._viewer_state.y_min = smallest_y_min

        self.redraw()

    @defer_draw
    def _calculate_profile_error(self, exc):
        self.plot_artist.set_visible(False)
        self.notify_end_computation()
        self.redraw()
        if issubclass(exc[0], IncompatibleAttribute):
            if isinstance(self.state.layer, BaseData):
                self.disable_invalid_attributes(self.state.attribute)
            else:
                self.disable_incompatible_subset()
        elif issubclass(exc[0], IncompatibleDataException):
            self.disable("Incompatible data")

    @defer_draw
    def _update_visual_attributes(self):

        if not self.enabled:
            return

        for mpl_artist in self.mpl_artists:
            mpl_artist.set_visible(self.state.visible)
            mpl_artist.set_zorder(self.state.zorder)
            mpl_artist.set_color(self.state.color)
            mpl_artist.set_alpha(self.state.alpha)
            mpl_artist.set_linewidth(self.state.linewidth)

        self.redraw()

    def _update_profile(self, force=True, **kwargs):

        if (self._viewer_state.x_att_pixel is None or
                self.state.attribute is None or
                self.state.layer is None):
            return

        changed = set() if force else self.pop_changed_properties()

        if force or any(prop in changed for prop in ('layer', 'slices', 'x_att', 'x_att_pixel', 'attribute',
                                                     'function', 'normalize', 'v_min', 'v_max', 'visible')):
            self._calculate_profile(reset=force)

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'linewidth')):
            self._update_visual_attributes()

    @defer_draw
    def update(self):
        self.state.reset_cache()
        self._update_profile(force=True)
        self.redraw()

    def get_handle_legend(self):
        if self.enabled and self.state.visible:
            handle = Line2D([0], [0], alpha=self.state.alpha,
                            linestyle="-", linewidth=self.state.linewidth,
                            color=self.get_layer_color())

            return handle, self.layer.label, None
        else:
            return None, None, None
