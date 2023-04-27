from collections import OrderedDict
from glue.core.hub import HubListener

import numpy as np

from glue.core import Subset
from echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.data_combo_helper import ManualDataComboHelper, ComponentIDComboHelper
from glue.utils import defer_draw, avoid_circular
from glue.core.data import BaseData
from glue.core.link_manager import is_convertible_to_single_pixel_cid
from glue.core.exceptions import IncompatibleDataException
from glue.core.message import SubsetUpdateMessage
from glue.core.units import find_unit_choices, UnitConverter

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

    x_display_unit = DDSCProperty(docstring='The units to use to display the x-axis.')
    y_display_unit = DDSCProperty(docstring='The units to use to display the y-axis')

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
        self.add_callback('reference_data', self._reference_data_changed, echo_old=True)
        self.add_callback('x_att', self._update_att)
        self.add_callback('x_display_unit', self._convert_units_x_limits, echo_old=True)
        self.add_callback('y_display_unit', self._convert_units_y_limits, echo_old=True)
        self.add_callback('normalize', self._reset_y_limits)
        self.add_callback('function', self._reset_y_limits)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att',
                                                   numeric=False, datetime=False, categorical=False,
                                                   pixel_coord=True)

        ProfileViewerState.function.set_choices(self, list(FUNCTIONS))
        ProfileViewerState.function.set_display_func(self, FUNCTIONS.get)

        def format_unit(unit):
            if unit is None:
                return 'Native units'
            else:
                return unit

        ProfileViewerState.x_display_unit.set_display_func(self, format_unit)
        ProfileViewerState.y_display_unit.set_display_func(self, format_unit)

        self.update_from_dict(kwargs)

    def _convert_units_x_limits(self, old_unit, new_unit):

        if old_unit != new_unit and self.reference_data is not None:

            limits = np.array([self.x_min, self.x_max])

            converter = UnitConverter()

            limits_native = converter.to_native(self.reference_data,
                                                self.x_att, limits,
                                                old_unit)

            limits_new = converter.to_unit(self.reference_data,
                                           self.x_att, limits_native,
                                           new_unit)

            with delay_callback(self, 'x_min', 'x_max'):
                self.x_min, self.x_max = sorted(limits_new)

    def _convert_units_y_limits(self, old_unit, new_unit):

        if old_unit != new_unit:

            if old_unit is None or new_unit is None:
                self._reset_y_limits()
                return

            limits = np.array([self.y_min, self.y_max])

            converter = UnitConverter()

            # We can use any layer, just find the first one that works and
            # exit if none of them do.
            for layer_state in self.layers:
                try:
                    layer_state.profile
                except Exception:  # e.g. incompatible subset
                    continue
                else:
                    break
            else:
                return

            if isinstance(layer_state.layer, Subset):
                data = layer_state.layer.data
            else:
                data = layer_state.layer

            limits_native = converter.to_native(data,
                                                layer_state.attribute, limits,
                                                old_unit)

            limits_new = converter.to_unit(self.reference_data,
                                           layer_state.attribute, limits_native,
                                           new_unit)

            with delay_callback(self, 'y_min', 'y_max'):
                self.y_min, self.y_max = sorted(limits_new)

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
        self._update_x_display_unit_choices()

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

        converter = UnitConverter()
        x_min, x_max = converter.to_unit(self.reference_data,
                                         self.x_att, np.array([x_min, x_max]),
                                         self.x_display_unit)

        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min = x_min
            self.x_max = x_max

    def _reset_y_limits(self, *event):
        if self.normalize:
            with delay_callback(self, 'y_min', 'y_max'):
                self.y_min = -0.1
                self.y_max = +1.1
        else:
            y_min, y_max = np.inf, -np.inf
            for layer in self.layers:
                try:
                    profile = layer.profile
                except Exception:  # e.g. incompatible subset
                    continue
                if profile is not None:
                    x, y = profile
                    if len(y) > 0:
                        y_min = min(y_min, np.nanmin(y))
                        y_max = max(y_max, np.nanmax(y))
            with delay_callback(self, 'y_min', 'y_max'):
                if y_max > y_min:
                    self.y_min = y_min
                    self.y_max = y_max
                else:
                    self.y_min = 0
                    self.y_max = 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_min, self.x_max = self.x_max, self.x_min

    @defer_draw
    @avoid_circular
    def _layers_changed(self, *args):
        # By default if any of the state properties change, this triggers a
        # callback on anything listening to changes on self.layers - but here
        # we just want to know if any layers have been removed/added so we keep
        # track of the UUIDs of the layers and check this before continuing.
        current_layers = [layer_state.layer.uuid for layer_state in self.layers]
        if not hasattr(self, '_last_layers') or self._last_layers != current_layers:
            self._update_combo_ref_data()
            self._update_y_display_unit_choices()
            self._last_layers = current_layers

    def _update_x_display_unit_choices(self):

        if self.reference_data is None:
            ProfileViewerState.x_display_unit.set_choices(self, [])
            return

        component = self.reference_data.get_component(self.x_att)
        if component.units:
            x_choices = find_unit_choices([(self.reference_data, self.x_att, component.units)])
        else:
            x_choices = ['']
        ProfileViewerState.x_display_unit.set_choices(self, x_choices)
        self.x_display_unit = component.units

    def _update_y_display_unit_choices(self):

        component_units = set()
        for layer_state in self.layers:
            if isinstance(layer_state.layer, BaseData):
                component = layer_state.layer.get_component(layer_state.attribute)
                if component.units:
                    component_units.add((layer_state.layer, layer_state.attribute, component.units))
        y_choices = [None] + find_unit_choices(component_units)
        ProfileViewerState.y_display_unit.set_choices(self, y_choices)

    @defer_draw
    def _reference_data_changed(self, before=None, after=None):

        # A callback event for reference_data is triggered if the choices change
        # but the actual selection doesn't - so we avoid resetting the WCS in
        # this case.
        if before is after:
            return

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

        self.reset_limits()

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name == 'reference_data':
            return 1.5
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1


class ProfileLayerState(MatplotlibLayerState, HubListener):
    """
    A state class that includes all the attributes for layers in a Profile plot.
    """

    linewidth = DDCProperty(1, docstring='The width of the line')

    attribute = DDSCProperty(docstring='The attribute shown in the layer')
    v_min = DDCProperty(docstring='The lower level shown')
    v_max = DDCProperty(docstring='The upper level shown')
    percentile = DDSCProperty(docstring='The percentile value used to '
                                        'automatically calculate levels')

    as_steps = DDCProperty(True, docstring='Whether to display the profile as steps')

    _viewer_callbacks_set = False
    _layer_subset_updates_subscribed = False
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

        self.add_callback('layer', self._on_layer_change, priority=1000)
        self.add_callback('visible', self.reset_cache, priority=1000)

        if layer is not None:
            self._on_layer_change()

        self.update_from_dict(kwargs)

    def _on_layer_change(self, *args):

        if self.layer is not None:

            # Set the available attributes
            self.attribute_att_helper.set_multiple_data([self.layer])

            # We only subscribe to SubsetUpdateMessage the first time that 'layer'
            # is not None, and then do any filtering in the callback function.
            if not self._layer_subset_updates_subscribed and self.layer.hub is not None:
                self.layer.hub.subscribe(self, SubsetUpdateMessage, handler=self._on_subset_update)
                self._layer_subset_updates_subscribed = True

        self.reset_cache()

    def _on_subset_update(self, msg):
        if msg.subset is self.layer:
            self.reset_cache()

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
            self.viewer_state.add_callback('x_display_unit', self.reset_cache, priority=100000)
            self.viewer_state.add_callback('y_display_unit', self.reset_cache, priority=100000)
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

            converter = UnitConverter()
            axis_values = converter.to_unit(self.viewer_state.reference_data,
                                            self.viewer_state.x_att, axis_values,
                                            self.viewer_state.x_display_unit)
            profile_values = converter.to_unit(data, self.attribute, profile_values,
                                               self.viewer_state.y_display_unit)

            self._profile_cache = axis_values, profile_values

        if update_limits:
            self.update_limits(update_profile=False)

    def update_limits(self, update_profile=True):
        with delay_callback(self, 'v_min', 'v_max'):
            if update_profile:
                self.update_profile(update_limits=False)
            if self._profile_cache is not None and len(self._profile_cache[1]) > 0:
                self.v_min = np.nanmin(self._profile_cache[1])
                self.v_max = np.nanmax(self._profile_cache[1])
