import numpy as np

from echo import (CallbackProperty, SelectionCallbackProperty,
                  delay_callback, ListCallbackProperty)
from glue.core.state_objects import StateAttributeLimitsHelper
from glue.viewers.common.state import ViewerState

__all__ = ['ViewerState3D']


class ViewerState3D(ViewerState):
    """
    A common state object for all 3D viewers
    """

    x_att = SelectionCallbackProperty()
    x_min = CallbackProperty(0)
    x_max = CallbackProperty(1)
    x_stretch = CallbackProperty(1.)

    y_att = SelectionCallbackProperty(default_index=1)
    y_min = CallbackProperty(0)
    y_max = CallbackProperty(1)
    y_stretch = CallbackProperty(1.)

    z_att = SelectionCallbackProperty(default_index=2)
    z_min = CallbackProperty(0)
    z_max = CallbackProperty(1)
    z_stretch = CallbackProperty(1.)

    visible_axes = CallbackProperty(True)
    perspective_view = CallbackProperty(False)
    clip_data = CallbackProperty(True)
    native_aspect = CallbackProperty(False)
    line_width = CallbackProperty(1.)

    layers = ListCallbackProperty()

    _limits_cache = CallbackProperty()

    def _update_priority(self, name):
        if name == 'layers':
            return 2
        elif name.endswith(('_min', '_max')):
            return 0
        else:
            return 1

    def __init__(self, **kwargs):

        super(ViewerState3D, self).__init__(**kwargs)

        if self._limits_cache is None:
            self._limits_cache = {}

        self.x_lim_helper = StateAttributeLimitsHelper(self, attribute='x_att',
                                                       lower='x_min', upper='x_max',
                                                       cache=self._limits_cache)

        self.y_lim_helper = StateAttributeLimitsHelper(self, attribute='y_att',
                                                       lower='y_min', upper='y_max',
                                                       cache=self._limits_cache)

        self.z_lim_helper = StateAttributeLimitsHelper(self, attribute='z_att',
                                                       lower='z_min', upper='z_max',
                                                       cache=self._limits_cache)

        # TODO: if limits_cache is re-assigned to a different object, we need to
        # update the attribute helpers. However if in future we make limits_cache
        # into a smart dictionary that can call callbacks when elements are
        # changed then we shouldn't always call this. It'd also be nice to
        # avoid this altogether and make it more clean.
        self.add_callback('_limits_cache', self._update_limits_cache)

    def reset_limits(self):
        self.x_lim_helper.log = False
        self.x_lim_helper.percentile = 100.
        self.x_lim_helper.update_values(force=True)
        self.y_lim_helper.log = False
        self.y_lim_helper.percentile = 100.
        self.y_lim_helper.update_values(force=True)
        self.z_lim_helper.log = False
        self.z_lim_helper.percentile = 100.
        self.z_lim_helper.update_values(force=True)

    def _update_limits_cache(self, *args):
        self.x_lim_helper._cache = self._limits_cache
        self.x_lim_helper._update_attribute()
        self.y_lim_helper._cache = self._limits_cache
        self.y_lim_helper._update_attribute()
        self.z_lim_helper._cache = self._limits_cache
        self.z_lim_helper._update_attribute()

    @property
    def aspect(self):
        # TODO: this could be cached based on the limits, but is not urgent
        aspect = np.array([1, 1, 1], dtype=float)
        if self.native_aspect:
            aspect[0] = 1.
            aspect[1] = (self.y_max - self.y_min) / (self.x_max - self.x_min)
            aspect[2] = (self.z_max - self.z_min) / (self.x_max - self.x_min)
            aspect /= aspect.max()
        return aspect

    def reset(self):
        pass

    def flip_x(self):
        self.x_lim_helper.flip_limits()

    def flip_y(self):
        self.y_lim_helper.flip_limits()

    def flip_z(self):
        self.z_lim_helper.flip_limits()

    @property
    def clip_limits(self):
        return (self.x_min, self.x_max,
                self.y_min, self.y_max,
                self.z_min, self.z_max)

    def set_limits(self, x_min, x_max, y_min, y_max, z_min, z_max):
        with delay_callback(self, 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max
            self.z_min = z_min
            self.z_max = z_max
