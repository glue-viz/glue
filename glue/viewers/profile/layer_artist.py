import sys
import warnings

from matplotlib.lines import Line2D


from glue.core import BaseData
from glue.utils import defer_draw
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

        drawstyle = 'steps-mid' if self.state.as_steps else 'default'
        self.plot_artist = self.axes.plot([1, 2, 3], [3, 4, 5], 'k-', drawstyle=drawstyle)[0]

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
            self.state.update_limits()
            # Normalize profile values to the [0:1] range based on limits
            if self._viewer_state.normalize:
                y = self.state.normalize_values(y)
            self.plot_artist.set_data(x, y)
        else:
            # We need to do this otherwise we get issues on Windows when
            # passing an empty list to plot_artist
            self.plot_artist.set_data([0.], [0.])

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
            mpl_artist.set_drawstyle('steps-mid' if self.state.as_steps else 'default')

        self.redraw()

    def _update_profile(self, force=False, **kwargs):

        if (self._viewer_state.x_att is None or
                self.state.attribute is None or
                self.state.layer is None):
            return

        # NOTE: we need to evaluate this even if force=True so that the cache
        # of updated properties is up to date after this method has been called.
        changed = self.pop_changed_properties()

        if force or any(prop in changed for prop in ('layer', 'x_att', 'attribute', 'function', 'normalize',
                                                     'v_min', 'v_max', 'visible', 'x_display_unit', 'y_display_unit')):
            self._calculate_profile(reset=force)
            force = True

        if force or any(prop in changed for prop in ('alpha', 'color', 'zorder', 'linewidth', 'as_steps')):
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
