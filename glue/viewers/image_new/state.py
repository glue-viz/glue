from __future__ import absolute_import, division, print_function

from glue.core import Data
from glue.config import colormaps
from glue.viewers.matplotlib.state import (MatplotlibDataViewerState,
                                           MatplotlibLayerState,
                                           DeferredDrawCallbackProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from astropy.visualization import LinearStretch

__all__ = ['ImageViewerState', 'ImageLayerState']


class ImageViewerState(MatplotlibDataViewerState):

    x_att = DeferredDrawCallbackProperty()
    y_att = DeferredDrawCallbackProperty()
    x_att_world = DeferredDrawCallbackProperty()
    y_att_world = DeferredDrawCallbackProperty()
    aspect = DeferredDrawCallbackProperty('equal')
    reference_data = DeferredDrawCallbackProperty()
    slices = DeferredDrawCallbackProperty()
    color_mode = DeferredDrawCallbackProperty('Colormaps')

    def __init__(self, **kwargs):

        super(ImageViewerState, self).__init__(**kwargs)

        self.add_callback('x_att_world', self._update_x_att)
        self.add_callback('y_att_world', self._update_y_att)

        self.limits_cache = {}

        self.x_att_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       limits_cache=self.limits_cache)

        self.y_att_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       limits_cache=self.limits_cache)

        self.add_callback('reference_data', self.set_default_slices)
        self.add_callback('layers', self.set_reference_data)

    def _update_x_att(self, *args):
        index = self.reference_data.world_component_ids.index(self.x_att_world)
        self.x_att = self.reference_data.pixel_component_ids[index]

    def _update_y_att(self, *args):
        index = self.reference_data.world_component_ids.index(self.y_att_world)
        self.y_att = self.reference_data.pixel_component_ids[index]

    def set_reference_data(self, *args):
        # TODO: make sure this doesn't get called for changes *in* the layers
        # for list callbacks maybe just have an event for length change in list
        if self.reference_data is None:
            for layer in self.layers:
                if isinstance(layer.layer, Data):
                    self.reference_data = layer.layer
                    return

    def set_default_slices(self, *args):
        # Need to make sure this gets called immediately when reference_data is changed
        if self.reference_data is None:
            self.slices = ()
        else:
            self.slices = (0,) * self.reference_data.ndim

    @property
    def numpy_slice_and_transpose(self):
        if self.reference_data is None:
            return None
        slices = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis or i == self.y_att.axis:
                slices.append(slice(None))
            else:
                slices.append(self.slices[i])
        transpose = self.y_att.axis > self.x_att.axis
        return slices, transpose

    @property
    def wcsaxes_slice(self):
        if self.reference_data is None:
            return None
        slices = []
        for i in range(self.reference_data.ndim):
            if i == self.x_att.axis:
                slices.append('x')
            elif i == self.y_att.axis:
                slices.append('y')
            else:
                slices.append(self.slices[i])
        return slices[::-1]

    def flip_x(self):
        self.x_att_helper.flip_limits()

    def flip_y(self):
        self.y_att_helper.flip_limits()


class ImageLayerState(MatplotlibLayerState):

    attribute = DeferredDrawCallbackProperty()
    v_min = DeferredDrawCallbackProperty()
    v_max = DeferredDrawCallbackProperty()
    percentile = DeferredDrawCallbackProperty(100)
    contrast = DeferredDrawCallbackProperty(1)
    bias = DeferredDrawCallbackProperty(0.5)
    cmap = DeferredDrawCallbackProperty()
    stretch = DeferredDrawCallbackProperty(LinearStretch)

    def __init__(self, **kwargs):
        super(ImageLayerState, self).__init__(**kwargs)
        self.attribute_helper = StateAttributeLimitsHelper(self, attribute='attribute',
                                                           percentile='percentile',
                                                           lower='v_min', upper='v_max')
        if self.cmap is None:
            self.cmap = colormaps.members[0][1]

    def flip_limits(self):
        self.attribute_helper.flip_limits()


class ImageSubsetLayerState(MatplotlibLayerState):
    pass
