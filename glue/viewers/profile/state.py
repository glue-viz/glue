from collections import OrderedDict

import numpy as np

from glue.core import Subset
from echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.data_combo_helper import ManualDataComboHelper, ComponentIDComboHelper
from glue.utils import defer_draw, nanmin, nanmax
from glue.core.link_manager import is_convertible_to_single_pixel_cid
from glue.core.exceptions import IncompatibleDataException

__all__ = ['ProfileViewerState', 'ProfileLayerState']


FUNCTIONS = OrderedDict([('maximum', 'Maximum'),
                         ('minimum', 'Minimum'),
                         ('mean', 'Mean'),
                         ('median', 'Median'),
                         ('sum', 'Sum')])


class ProfileViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a Profile viewer.
    """

    x_att_pixel = DDCProperty(docstring='The component ID giving the pixel component '
                                  'shown on the x axis')

    x_att = DDSCProperty(docstring='The component ID giving the pixel or world component '
                                   'shown on the x axis')

    reference_data = DDSCProperty(docstring='The dataset that is used to define the '
                                            'available pixel/world components, and '
                                            'which defines the coordinate frame in '
                                            'which the images are shown')

    function = DDSCProperty(docstring='The function to use for collapsing data')

    normalize = DDCProperty(False, docstring='Whether to normalize all profiles '
                                             'to the [0:1] range')

    # TODO: add function to use

    def __init__(self, **kwargs):

        super(ProfileViewerState, self).__init__()

        self.ref_data_helper = ManualDataComboHelper(self, 'reference_data')

        self.add_callback('layers', self._layers_changed)
        self.add_callback('reference_data', self._reference_data_changed)
        self.add_callback('x_att', self._update_att)
        self.add_callback('normalize', self._reset_y_limits)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att',
                                                   numeric=False, datetime=False, categorical=False,
                                                   pixel_coord=True)

        ProfileViewerState.function.set_choices(self, list(FUNCTIONS))
        ProfileViewerState.function.set_display_func(self, FUNCTIONS.get)

        self.update_from_dict(kwargs)

    def _update_combo_ref_data(self):
        self.ref_data_helper.set_multiple_data(self.layers_data)

    def reset_limits(self):
        with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max'):
            self._reset_x_limits()
            self._reset_y_limits()

    @property
    def _display_world(self):
        return getattr(self.reference_data, 'coords', None) is not None

    @defer_draw
    def _update_att(self, *args):
        if self.x_att is not None:
            if self._display_world:
                if self.x_att in self.reference_data.pixel_component_ids:
                    self.x_att_pixel = self.x_att
                else:
                    index = self.reference_data.world_component_ids.index(self.x_att)
                    self.x_att_pixel = self.reference_data.pixel_component_ids[index]
            else:
                self.x_att_pixel = self.x_att
        self._reset_x_limits()

    def _reset_x_limits(self, *event):

        # NOTE: we don't use AttributeLimitsHelper because we need to avoid
        # trying to get the minimum of *all* the world coordinates in the
        # dataset. Instead, we use the same approach as in the layer state below
        # and in the case of world coordinates we use online the spine of the
        # data.

        if self.reference_data is None or self.x_att_pixel is None:
            return

        data = self.reference_data

        if self.x_att in data.pixel_component_ids:
            x_min, x_max = -0.5, data.shape[self.x_att.axis] - 0.5
        else:
            axis = data.world_component_ids.index(self.x_att)
            axis_view = [0] * data.ndim
            axis_view[axis] = slice(None)
            axis_values = data[self.x_att, tuple(axis_view)]
            x_min, x_max = np.nanmin(axis_values), np.nanmax(axis_values)

        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min = x_min
            self.x_max = x_max

    def _reset_y_limits(self, *event):
        if self.normalize:
            with delay_callback(self, 'y_min', 'y_max'):
                self.y_min = -0.1
                self.y_max = +1.1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min, self.x_max = self.x_max, self.x_min

    @defer_draw
    def _layers_changed(self, *args):
        self._update_combo_ref_data()

    @defer_draw
    def _reference_data_changed(self, *args):

        for layer in self.layers:
            layer.reset_cache()

        # This signal can get emitted if just the choices but not the actual
        # reference data change, so we check here that the reference data has
        # actually changed
        if self.reference_data is not getattr(self, '_last_reference_data', None):
            self._last_reference_data = self.reference_data

            with delay_callback(self, 'x_att'):

                if self.reference_data is None:
                    self.x_att_helper.set_multiple_data([])
                else:
                    self.x_att_helper.set_multiple_data([self.reference_data])
                    if self._display_world:
                        self.x_att_helper.world_coord = True
                        self.x_att = self.reference_data.world_component_ids[0]
                    else:
                        self.x_att_helper.world_coord = False
                        self.x_att = self.reference_data.pixel_component_ids[0]

                self._update_att()

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name == 'reference_data':
            return 1.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1


class ProfileLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a Profile plot.
    """

    linewidth = DDCProperty(1, docstring='The width of the line')

    attribute = DDSCProperty(docstring='The attribute shown in the layer')
    v_min = DDCProperty(docstring='The lower level shown')
    v_max = DDCProperty(docstring='The upper level shown')
    percentile = DDSCProperty(docstring='The percentile value used to '
                                        'automatically calculate levels')

    _viewer_callbacks_set = False
    _profile_cache = None

    def __init__(self, layer=None, viewer_state=None, **kwargs):

        super(ProfileLayerState, self).__init__(layer=layer, viewer_state=viewer_state)

        self.attribute_att_helper = ComponentIDComboHelper(self, 'attribute',
                                                           numeric=True, categorical=False)

        percentile_display = {100: 'Min/Max',
                              99.5: '99.5%',
                              99: '99%',
                              95: '95%',
                              90: '90%',
                              'Custom': 'Custom'}

        ProfileLayerState.percentile.set_choices(self, [100, 99.5, 99, 95, 90, 'Custom'])
        ProfileLayerState.percentile.set_display_func(self, percentile_display.get)

        self.add_callback('layer', self._update_attribute, priority=1000)

        if layer is not None:
            self._update_attribute()

        self.update_from_dict(kwargs)

    def _update_attribute(self, *args):
        if self.layer is not None:
            self.attribute_att_helper.set_multiple_data([self.layer])

    @property
    def independent_x_att(self):
        return is_convertible_to_single_pixel_cid(self.layer, self.viewer_state.x_att) is not None

    def normalize_values(self, values):
        return (np.asarray(values) - self.v_min) / (self.v_max - self.v_min)

    def reset_cache(self, *args):
        self._profile_cache = None

    @property
    def viewer_state(self):
        return self._viewer_state

    @viewer_state.setter
    def viewer_state(self, viewer_state):
        self._viewer_state = viewer_state

    @property
    def profile(self):
        self.update_profile()
        return self._profile_cache

    def update_profile(self, update_limits=True):

        if self._profile_cache is not None:
            return self._profile_cache

        if not self.visible:
            return

        if not self._viewer_callbacks_set:
            self.viewer_state.add_callback('x_att', self.reset_cache, priority=100000)
            self.viewer_state.add_callback('function', self.reset_cache, priority=100000)
            if self.is_callback_property('attribute'):
                self.add_callback('attribute', self.reset_cache, priority=100000)
            self._viewer_callbacks_set = True

        if self.viewer_state is None or self.viewer_state.x_att is None or self.attribute is None:
            raise IncompatibleDataException()

        # Check what pixel axis in the current dataset x_att corresponds to
        pix_cid = is_convertible_to_single_pixel_cid(self.layer, self.viewer_state.x_att_pixel)

        if pix_cid is None:
            raise IncompatibleDataException()

        # If we get here, then x_att does correspond to a single pixel axis in
        # the cube, so we now prepare a list of axes to collapse over.
        axes = tuple(i for i in range(self.layer.ndim) if i != pix_cid.axis)

        # We now get the y values for the data

        # TODO: in future we should optimize the case where the mask is much
        # smaller than the data to just average the relevant 'spaxels' in the
        # data rather than collapsing the whole cube.

        if isinstance(self.layer, Subset):
            data = self.layer.data
            subset_state = self.layer.subset_state
        else:
            data = self.layer
            subset_state = None

        profile_values = data.compute_statistic(self.viewer_state.function, self.attribute, axis=axes, subset_state=subset_state)

        if np.all(np.isnan(profile_values)):
            self._profile_cache = [], []
        else:
            axis_view = [0] * data.ndim
            axis_view[pix_cid.axis] = slice(None)
            axis_values = data[self.viewer_state.x_att, tuple(axis_view)]
            self._profile_cache = axis_values, profile_values

        if update_limits:
            self.update_limits(update_profile=False)

    def update_limits(self, update_profile=True):
        with delay_callback(self, 'v_min', 'v_max'):
            if update_profile:
                self.update_profile(update_limits=False)
            if self._profile_cache is not None and len(self._profile_cache[1]) > 0:
                self.v_min = nanmin(self._profile_cache[1])
                self.v_max = nanmax(self._profile_cache[1])
