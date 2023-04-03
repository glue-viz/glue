from echo import CallbackProperty, SelectionCallbackProperty, keep_in_sync, delay_callback

from matplotlib.colors import to_rgba

from glue.core.message import LayerArtistUpdatedMessage

from glue.core.state_objects import State
from glue.viewers.common.state import ViewerState, LayerState

from glue.utils import defer_draw, avoid_circular

__all__ = ['DeferredDrawSelectionCallbackProperty', 'DeferredDrawCallbackProperty',
           'MatplotlibDataViewerState', 'MatplotlibLayerState']


class DeferredDrawCallbackProperty(CallbackProperty):
    """
    A callback property where drawing is deferred until
    after notify has called all callback functions.
    """

    @defer_draw
    def notify(self, *args, **kwargs):
        super(DeferredDrawCallbackProperty, self).notify(*args, **kwargs)


class DeferredDrawSelectionCallbackProperty(SelectionCallbackProperty):
    """
    A callback property where drawing is deferred until
    after notify has called all callback functions.
    """

    @defer_draw
    def notify(self, *args, **kwargs):
        super(DeferredDrawSelectionCallbackProperty, self).notify(*args, **kwargs)


VALID_WEIGHTS = ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']


VALID_LOCATIONS = ['draggable', 'best',
                   'upper right', 'upper left',
                   'lower left', 'lower right',
                   'center left', 'center right',
                   'lower center', 'upper center']


class MatplotlibLegendState(State):
    """The legend state"""

    visible = DeferredDrawCallbackProperty(False, docstring="Whether to show the legend")

    location = DeferredDrawSelectionCallbackProperty(0, docstring="The location of the legend in the axis")

    title = DeferredDrawCallbackProperty("", docstring='The title of the legend')
    fontsize = DeferredDrawCallbackProperty(10, docstring='The font size of the title')

    alpha = DeferredDrawCallbackProperty(0.6, docstring='Transparency of the legend frame')
    frame_color = DeferredDrawCallbackProperty("#ffffff", docstring='Frame color of the legend')
    show_edge = DeferredDrawCallbackProperty(True, docstring="Whether to show the edge of the frame ")
    text_color = DeferredDrawCallbackProperty("#000000", docstring='Text color of the legend')

    def __init__(self, *args, **kwargs):
        MatplotlibLegendState.location.set_choices(self, VALID_LOCATIONS)

        super().__init__(*args, **kwargs)
        self._set_color_choices()

    def _set_color_choices(self):
        from glue.config import settings

        self.frame_color = settings.BACKGROUND_COLOR
        self.text_color = settings.FOREGROUND_COLOR

    @property
    def edge_color(self):
        if self.show_edge:
            return to_rgba(self.text_color, self.alpha)
        else:
            return None

    @property
    def draggable(self):
        return self.location == 'draggable'

    @property
    def mpl_location(self):
        if self.location == 'draggable':
            return 'best'
        else:
            return self.location

    def update_axes_settings_from(self, state):
        self.visible = state.show_legend
        self.loc_and_drag = state.loc_and_drag
        self.alpha = state.alpha
        self.title = state.title
        self.fontsize = state.fontsize
        self.frame_color = state.frame_color
        self.show_edge = state.show_edge
        self.text_color = state.text_color


class MatplotlibDataViewerState(ViewerState):
    """
    A base class that includes common attributes for viewers based on
    Matplotlib.
    """

    x_min = DeferredDrawCallbackProperty(docstring='Lower limit of the visible x range')
    x_max = DeferredDrawCallbackProperty(docstring='Upper limit of the visible x range')

    y_min = DeferredDrawCallbackProperty(docstring='Lower limit of the visible y range')
    y_max = DeferredDrawCallbackProperty(docstring='Upper limit of the visible y range')

    x_log = DeferredDrawCallbackProperty(False, docstring='Whether the x axis is logarithmic')
    y_log = DeferredDrawCallbackProperty(False, docstring='Whether the y axis is logarithmic')

    aspect = DeferredDrawCallbackProperty('auto', docstring='Aspect ratio for the axes')

    show_axes = DeferredDrawCallbackProperty(True, docstring='Whether the axes are shown')

    x_axislabel = DeferredDrawCallbackProperty('', docstring='Label for the x-axis')
    y_axislabel = DeferredDrawCallbackProperty('', docstring='Label for the y-axis')

    x_axislabel_size = DeferredDrawCallbackProperty(10, docstring='Size of the x-axis label')
    y_axislabel_size = DeferredDrawCallbackProperty(10, docstring='Size of the y-axis label')

    x_axislabel_weight = DeferredDrawSelectionCallbackProperty(1, docstring='Weight of the x-axis label')
    y_axislabel_weight = DeferredDrawSelectionCallbackProperty(1, docstring='Weight of the y-axis label')

    x_ticklabel_size = DeferredDrawCallbackProperty(8, docstring='Size of the x-axis tick labels')
    y_ticklabel_size = DeferredDrawCallbackProperty(8, docstring='Size of the y-axis tick labels')

    def __init__(self, *args, **kwargs):

        self._axes_aspect_ratio = None

        MatplotlibDataViewerState.x_axislabel_weight.set_choices(self, VALID_WEIGHTS)
        MatplotlibDataViewerState.y_axislabel_weight.set_choices(self, VALID_WEIGHTS)

        super(MatplotlibDataViewerState, self).__init__(*args, **kwargs)
        self.legend = MatplotlibLegendState(*args, **kwargs)

        self.add_callback('aspect', self._adjust_limits_aspect, priority=10000)
        self.add_callback('x_min', self._adjust_limits_aspect_x, priority=10000)
        self.add_callback('x_max', self._adjust_limits_aspect_x, priority=10000)
        self.add_callback('y_min', self._adjust_limits_aspect_y, priority=10000)
        self.add_callback('y_max', self._adjust_limits_aspect_y, priority=10000)

    def _set_axes_aspect_ratio(self, value):
        """
        Set the aspect ratio of the axes in which the visualization is shown.
        This is a private method that is intended only for internal use, and it
        allows this viewer state class to adjust the limits accordingly when
        the aspect callback property is set to 'equal'
        """
        self._axes_aspect_ratio = value
        self._adjust_limits_aspect(aspect_adjustable='both')

    def _adjust_limits_aspect_x(self, *args):
        self._adjust_limits_aspect(aspect_adjustable='y')

    def _adjust_limits_aspect_y(self, *args):
        self._adjust_limits_aspect(aspect_adjustable='x')

    @avoid_circular
    def _adjust_limits_aspect(self, *args, **kwargs):
        """
        Adjust the limits of the visualization to take into account the aspect
        ratio. This only works if `_set_axes_aspect_ratio` has been called
        previously.
        """

        if self.aspect == 'auto' or self._axes_aspect_ratio is None:
            return

        if self.x_min is None or self.x_max is None or self.y_min is None or self.y_max is None:
            return

        aspect_adjustable = kwargs.pop('aspect_adjustable', 'auto')

        changed = None

        # Find axes aspect ratio
        axes_ratio = self._axes_aspect_ratio

        # Put the limits in temporary variables so that we only actually change
        # them in one go at the end.
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        # Find current data ratio
        data_ratio = abs(y_max - y_min) / abs(x_max - x_min)

        # Only do something if the data ratio is sufficiently different
        # from the axes ratio.
        if abs(data_ratio - axes_ratio) / (0.5 * (data_ratio + axes_ratio)) > 0.0001:

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

            elif (aspect_adjustable == 'auto' and data_ratio > axes_ratio) or aspect_adjustable == 'x':
                x_mid = 0.5 * (x_min + x_max)
                x_width = abs(y_max - y_min) / axes_ratio
                x_min = x_mid - x_width / 2.
                x_max = x_mid + x_width / 2.
            else:
                y_mid = 0.5 * (y_min + y_max)
                y_width = abs(x_max - x_min) * axes_ratio
                y_min = y_mid - y_width / 2.
                y_max = y_mid + y_width / 2.

            with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max'):
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max

    def update_axes_settings_from(self, state):
        # axis
        self.x_axislabel_size = state.x_axislabel_size
        self.y_axislabel_size = state.y_axislabel_size
        self.x_axislabel_weight = state.x_axislabel_weight
        self.y_axislabel_weight = state.y_axislabel_weight
        self.x_ticklabel_size = state.x_ticklabel_size
        self.y_ticklabel_size = state.y_ticklabel_size
        # legend
        self.legend.update_axes_settings_from(state.legend)

    @defer_draw
    def _notify_global(self, *args, **kwargs):
        super(MatplotlibDataViewerState, self)._notify_global(*args, **kwargs)

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith('_log'):
            return 0.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def __gluestate__(self, context):
        state = super(MatplotlibDataViewerState, self).__gluestate__(context)
        state["legend"] = self.legend.as_dict()
        return state

    @classmethod
    def __setgluestate__(cls, rec, context):
        state = super(MatplotlibDataViewerState, cls).__setgluestate__(rec, context)
        if "legend" in rec:
            state.legend.update_from_dict(rec["legend"])
        return state


class MatplotlibLayerState(LayerState):
    """
    A base class that includes common attributes for all layers in viewers based
    on Matplotlib.
    """

    color = DeferredDrawCallbackProperty(docstring='The color used to display '
                                                   'the data')
    alpha = DeferredDrawCallbackProperty(docstring='The transparency used to '
                                                   'display the data')

    def __init__(self, viewer_state=None, **kwargs):

        super(MatplotlibLayerState, self).__init__(viewer_state=viewer_state, **kwargs)

        self.color = self.layer.style.color
        self.alpha = self.layer.style.alpha

        self._sync_color = keep_in_sync(self, 'color', self.layer.style, 'color')
        self._sync_alpha = keep_in_sync(self, 'alpha', self.layer.style, 'alpha')

        self.add_global_callback(self._notify_layer_update)

    def _notify_layer_update(self, **kwargs):
        message = LayerArtistUpdatedMessage(self)
        if self.layer is not None and self.layer.hub is not None:
            self.layer.hub.broadcast(message)

    @defer_draw
    def _notify_global(self, *args, **kwargs):
        super(MatplotlibLayerState, self)._notify_global(*args, **kwargs)
