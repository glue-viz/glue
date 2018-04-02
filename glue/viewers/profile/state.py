from __future__ import absolute_import, division, print_function

import numpy as np

from glue.core import Data
from glue.external.echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.utils import defer_draw
from glue.core.link_manager import is_convertible_to_single_pixel_cid
from glue.core.exceptions import IncompatibleAttribute

__all__ = ['ProfileViewerState', 'ProfileLayerState']


FUNCTIONS = {np.nanmean: 'Mean',
             np.nanmedian: 'Median',
             np.nanmin: 'Minimum',
             np.nanmax: 'Maximum',
             np.nansum: 'Sum'}


class ProfileViewerState(MatplotlibDataViewerState):
    """
    A state class that includes all the attributes for a Profile viewer.
    """

    x_att = DDSCProperty(docstring='The data component to use for the x-axis '
                                   'of the profile (should be a pixel component)')

    y_att = DDSCProperty(docstring='The data component to use for the y-axis '
                                       'of the profile')

    function = DDSCProperty(docstring='The function to use for collapsing data')

    # TODO: add function to use

    def __init__(self, **kwargs):

        super(ProfileViewerState, self).__init__()

        self.x_lim_helper = StateAttributeLimitsHelper(self, 'x_att', lower='x_min',
                                                       upper='x_max')

        self.add_callback('layers', self._layers_changed)

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att',
                                                   numeric=False, categorical=False,
                                                   world_coord=True, pixel_coord=True)
        self.y_att_helper = ComponentIDComboHelper(self, 'y_att', numeric=True)

        ProfileViewerState.function.set_choices(self, list(FUNCTIONS))
        ProfileViewerState.function.set_display_func(self, FUNCTIONS.get)

        self.update_from_dict(kwargs)

    def reset_limits(self):
        with delay_callback(self, 'x_min', 'x_max'):
            self.x_lim_helper.percentile = 100
            self.x_lim_helper.update_values(force=True)

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
        self.x_att_helper.set_multiple_data(self.layers_data)
        self.y_att_helper.set_multiple_data(self.layers_data)


class ProfileLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a Profile plot.
    """

    linewidth = DDCProperty(1, docstring='The width of the line')

    @property
    def independent_x_att(self):
        return is_convertible_to_single_pixel_cid(self.layer, self.viewer_state.x_att) is not None

    def get_profile(self):

        # Check what pixel axis in the current dataset x_att corresponds to
        pix_cid = is_convertible_to_single_pixel_cid(self.layer, self.viewer_state.x_att)

        if pix_cid is None:
            raise IncompatibleAttribute()

        # If we get here, then x_att does correspond to a single pixel axis in
        # the cube, so we now prepare a list of axes to collapse over.
        axes = tuple(i for i in range(self.layer.ndim) if i != pix_cid.axis)

        # We now get the y values for the data

        # TODO: in future we should optimize the case where the mask is much
        # smaller than the data to just average the relevant 'spaxels' in the
        # data rather than collapsing the whole cube.

        if isinstance(self.layer, Data):
            data = self.layer
            data_values = data[self.viewer_state.y_att]
        else:
            # We need to force a copy *and* convert to float just in case
            data = self.layer.data
            data_values = np.array(data[self.viewer_state.y_att], dtype=float)
            mask = self.layer.to_mask()
            if np.sum(mask) == 0:
                return [], []
            data_values[~mask] = np.nan

        # Collapse along all dimensions except x_att
        profile_values = self.viewer_state.function(data_values, axis=axes)
        profile_values[np.isnan(profile_values)] = 0.

        # Finally, we get the coordinate values for the requested axis
        axis_view = [0] * data.ndim
        axis_view[pix_cid.axis] = slice(None)
        axis_values = data[self.viewer_state.x_att, axis_view]

        return axis_values, profile_values
