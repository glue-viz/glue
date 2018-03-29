from __future__ import absolute_import, division, print_function

import numpy as np

from glue.external.echo import delay_callback
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty as DDCProperty,
                                           DeferredDrawSelectionCallbackProperty as DDSCProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.core.data_combo_helper import ComponentIDComboHelper
from glue.utils import defer_draw

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

        self.x_att_helper = ComponentIDComboHelper(self, 'x_att', numeric=False, categorical=False, pixel_coord=True)
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
        print("X_ATT [bef]", self.x_att)
        print("Y_ATT [bef]", self.y_att)
        self.x_att_helper.set_multiple_data(self.layers_data)
        self.y_att_helper.set_multiple_data(self.layers_data)
        print("X_ATT [aft]", self.x_att)
        print("Y_ATT [aft]", self.y_att)


class ProfileLayerState(MatplotlibLayerState):
    """
    A state class that includes all the attributes for layers in a Profile plot.
    """

    linewidth = DDCProperty(1, docstring='The width of the line')
