from __future__ import absolute_import, division, print_function

import warnings
from collections import OrderedDict

import numpy as np

from glue.core import Data, Subset, Coordinates
from glue.external.echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.core.data_combo_helper import ManualDataComboHelper, ComponentIDComboHelper
from glue.utils import defer_draw, nanmean, nanmedian, nansum, nanmin, nanmax, iterate_chunks
from glue.core.link_manager import is_convertible_to_single_pixel_cid
from glue.core.exceptions import IncompatibleDataException
from glue.core.subset import SliceSubsetState

__all__ = ['ProfileViewerState', 'ProfileLayerState']


FUNCTIONS = OrderedDict([(nanmax, 'Maximum'),
                         (nanmin, 'Minimum'),
                         (nanmean, 'Mean'),
                         (nanmedian, 'Median'),
                         (nansum, 'Sum')])

# Maximum number of elements in a chunk size used to compute the profile - this
# prevents the profile calculation from using up too much memory at a time.
N_CHUNK_MAX = 50000000


class ProfileViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a Profile viewer.
    """

    reference_data = DDSCProperty(docstring='The dataset that is used to define the '
                                            'available pixel/world components, and '
                                            'which defines the coordinate frame in '
                                            'which the images are shown')

    x_att = DDSCProperty(docstring='The data component to use for the x-axis '
                                   'of the profile (should be a pixel component)')

    function = DDSCProperty(docstring='The function to use for collapsing data')

    normalize = DDCProperty(False, docstring='Whether to normalize all profiles '
                                             'to the [0:1] range')

    # TODO: add function to use

    def __init__(self, **kwargs):

        super(ProfileViewerState, self).__init__()

        self.ref_data_helper = ManualDataComboHelper(self, 'reference_data')

        self.x_lim_helper = StateAttributeLimitsHelper(self, 'x_att', lower='x_min',
                                                       upper='x_max')

        self.add_callback('layers', self._layers_changed)
        self.add_callback('reference_data', self._reference_data_changed)
        self.add_callback('normalize', self._reset_y_limits)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att',
                                                   numeric=False, categorical=False,
                                                   world_coord=True, pixel_coord=True)

        ProfileViewerState.function.set_choices(self, list(FUNCTIONS))
        ProfileViewerState.function.set_display_func(self, FUNCTIONS.get)

        self.update_from_dict(kwargs)

    def _update_combo_ref_data(self):
        self.ref_data_helper.set_multiple_data(self.layers_data)

    def reset_limits(self):
        with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max'):
            self.x_lim_helper.percentile = 100
            self.x_lim_helper.update_values(force=True)
            self._reset_y_limits()

    def _reset_y_limits(self, *event):
        if self.normalize:
            self.y_min = -0.1
            self.y_max = +1.1

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def flip_x(self):
        """
        Flip the x_min/x_max limits.
        """
        self.x_lim_helper.flip_limits()

    @defer_draw
    def _layers_changed(self, *args):
        self._update_combo_ref_data()

    @defer_draw
    def _reference_data_changed(self, *args):
        if self.reference_data is None:
            self.x_att_helper.set_multiple_data([])
        else:
            self.x_att_helper.set_multiple_data([self.reference_data])
            if type(self.reference_data.coords) == Coordinates:
                self.x_att = self.reference_data.pixel_component_ids[0]
            else:
                self.x_att = self.reference_data.world_component_ids[0]


class ProfileLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a Profile plot.
    """

    linewidth = DDCProperty(1, docstring='The width of the line')

    attribute = DDSCProperty(docstring='The attribute shown in the layer')
    v_min = DDCProperty(docstring='The lower level shown')
    v_max = DDCProperty(docstring='The upper leven shown')
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

        if not self._viewer_callbacks_set:
            self.viewer_state.add_callback('x_att', self.reset_cache, priority=100000)
            self.viewer_state.add_callback('function', self.reset_cache, priority=100000)
            if self.is_callback_property('attribute'):
                self.add_callback('attribute', self.reset_cache, priority=100000)
            self._viewer_callbacks_set = True

        if self.viewer_state is None or self.viewer_state.x_att is None or self.attribute is None:
            raise IncompatibleDataException()

        # Check what pixel axis in the current dataset x_att corresponds to
        pix_cid = is_convertible_to_single_pixel_cid(self.layer, self.viewer_state.x_att)

        if pix_cid is None:
            raise IncompatibleDataException()

        # If we get here, then x_att does correspond to a single pixel axis in
        # the cube, so we now prepare a list of axes to collapse over.
        axes = tuple(i for i in range(self.layer.ndim) if i != pix_cid.axis)

        # We now get the y values for the data

        # TODO: in future we should optimize the case where the mask is much
        # smaller than the data to just average the relevant 'spaxels' in the
        # data rather than collapsing the whole cube.

        # We operate in chunks here to avoid memory issues

        axis_index = pix_cid.axis

        profile_values = np.zeros(self.layer.shape[axis_index])

        if isinstance(self.layer, Subset) and isinstance(self.layer.subset_state, SliceSubsetState):
            chunk_shape = self.layer.shape
        else:
            chunk_shape = list(self.layer.shape)
            if self.layer.size > N_CHUNK_MAX:
                # Deliberately leave n_chunks as float to not round twice
                n_chunks = self.layer.size / N_CHUNK_MAX
                chunk_shape[axis_index] = max(1, int(chunk_shape[axis_index] / n_chunks))

        # TODO: there are cases where the code below is not optimized because
        # the mask may be computable for a single slice and broadcastable to all
        # slices - normally ROISubsetState takes care of that but if we call it
        # once per view it won't. In the future we could ask a SubsetState
        # whether it is broadcasted along axis_index.

        for view in iterate_chunks(self.layer.shape, chunk_shape=chunk_shape):

            if isinstance(self.layer, Data):
                data = self.layer
                data_values = data[self.attribute, view]
            else:
                data = self.layer.data
                if isinstance(self.layer.subset_state, SliceSubsetState):
                    data_values = self.layer.subset_state.to_array(self.layer.data, self.attribute)
                else:
                    # We need to force a copy *and* convert to float just in case
                    data_values = np.array(data[self.attribute, view], dtype=float)
                    mask = self.layer.to_mask(view=view)
                    data_values[~mask] = np.nan

            # Collapse along all dimensions except x_att
            if self.layer.ndim > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    profile_values[view[axis_index]] = self.viewer_state.function(data_values, axis=axes)
            else:
                profile_values[view[axis_index]] = data_values

        # Finally, we get the coordinate values for the requested axis
        if np.all(np.isnan(profile_values)):
            self._profile_cache = [], []
        else:
            axis_view = [0] * data.ndim
            axis_view[pix_cid.axis] = slice(None)
            axis_values = data[self.viewer_state.x_att, axis_view]
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
