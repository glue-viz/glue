from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
from collections import Counter, defaultdict

import numpy as np

from glue.utils import nonpartial
from glue.core import Subset
from glue.external import six
from glue.external.echo import add_callback, CallbackProperty, ignore_callback, ListCallbackProperty, HasCallbackProperties


__all__ = ['State']


class State(HasCallbackProperties):
    """
    A class to represent the state of a UI element. Initially this doesn't add
    anything compared to HasCallbackProperties, but functionality will be added
    over time.
    """

    def __init__(self, **kwargs):
        super(State, self).__init__()
        for name in kwargs:
            if self.is_property(name):
                setattr(self, name, kwargs[name])

    def is_property(self, name):
        return isinstance(getattr(type(self), name, None), CallbackProperty)


class StateAttributeLimitsHelper(object):
    """
    This class is a helper for attribute-dependent min/max level values. It
    is equivalent to AttributeLimitsHelper but operates on State objects and
    is GUI-independent.

    Parameters
    ----------
    attribute : str
        The attribute name - this will be populated once a dataset is assigned
        to the helper.
    lower, upper : str
        The fields for the lower/upper levels
    mode : ``QComboBox`` instance, optional
        The scale mode combo - this will be populated by presets such as
        Min/Max, various percentile levels, and Custom.
    log_button : ``QToolButton`` instance, optional
        A button indicating whether the attribute should be shown in log space

    Notes
    -----

    Once the helper is instantiated, the data associated with the helper can be
    set/changed with:

    >>> helper = AttributeLimitsHelper(...)
    >>> helper.data = data

    The data can also be passed to the initializer as described in the list of
    parameters above.
    """

    def __init__(self, state, attribute, vlo, vhi,
                 percentile=None, vlog=None, limits_cache=None):

        self._state = state

        self._attribute = getattr(type(state), attribute)
        self._vlo = getattr(type(state), vlo)
        self._vhi = getattr(type(state), vhi)

        self._attribute.add_callback(self._state, nonpartial(self._update_limits))
        self._vlo.add_callback(self._state, nonpartial(self._manual_edit))
        self._vhi.add_callback(self._state, nonpartial(self._manual_edit))

        if vlog is not None:
            self._vlog = getattr(type(state), vlog)
            self._vlog.add_callback(self._state, nonpartial(self._manual_edit))
        else:
            self._vlog = CallbackProperty()

        if percentile is not None:
            self._percentile = getattr(type(state), percentile)
            self._percentile.add_callback(self._state, nonpartial(self._update_percentile))
        else:
            self._percentile = CallbackProperty()

        if limits_cache is None:
            limits_cache = {}

        self._limits = limits_cache

        if self.data is not None:
            self._update_data()
            self._update_limits()

    def _update_data(self):
        if self.attribute is None or self.attribute[1] is not self.data:
            if isinstance(self.data, Subset):
                self.attribute = self.data.data.visible_components[0], self.data.data
            else:
                self.attribute = self.data.visible_components[0], self.data

    def set_limits(self, vlo, vhi):
        # FIXME: delay so that both notifications go out at the same time
        self.vlo = vlo
        self.vhi = vhi

    def flip_limits(self):
        self.set_limits(self.vhi, self.vlo)

    def _manual_edit(self):
        self._cache_limits()

    def _update_percentile(self):
        if self.percentile is not None:
            self._auto_limits()
            self._cache_limits()

    def _invalidate_cache(self):
        self._limits.clear()

    def _cache_limits(self):
        self._limits[self.component_id] = self.percentile, self.vlo, self.vhi, self.vlog

    def _update_limits(self):
        if self.component_id in self._limits:
            self.percentile, lower, upper, self.vlog = self._limits[self.component_id]
            self.set_limits(lower, upper)
        else:
            # Block signals here?
            self.percentile = 100
            self._auto_limits()
            self.vlog = False

    def _auto_limits(self):

        if self.data is None:
            return

        exclude = (100 - self.percentile) / 2.

        # For subsets in 'data' mode, we want to compute the limits based on
        # the full dataset, not just the subset.
        if isinstance(self.data, Subset):
            data_values = self.data.data[self.component_id]
        else:
            data_values = self.data[self.component_id]

        try:
            lower = np.nanpercentile(data_values, exclude)
            upper = np.nanpercentile(data_values, 100 - exclude)
        except AttributeError:  # Numpy < 1.9
            data_values = data_values[~np.isnan(data_values)]
            lower = np.percentile(data_values, exclude)
            upper = np.percentile(data_values, 100 - exclude)

        if isinstance(self.data, Subset):
            lower = 0

        self.set_limits(lower, upper)

    # FIXME: We need to find a more elegant way to do the following!

    @property
    def data(self):
        if self.attribute is None:
            return None
        else:
            return self.attribute[1]

    @property
    def attribute(self):
        return self._attribute.__get__(self._state)

    @attribute.setter
    def attribute(self, value):
        return self._attribute.__set__(self._state, value)

    @property
    def component_id(self):
        if self.attribute is None:
            return None
        else:
            return self.attribute[0]

    @property
    def percentile(self):
        return self._percentile.__get__(self._state)

    @percentile.setter
    def percentile(self, value):
        return self._percentile.__set__(self._state, value)

    @property
    def vlo(self):
        return self._vlo.__get__(self._state)

    @vlo.setter
    def vlo(self, value):
        return self._vlo.__set__(self._state, value)

    @property
    def vhi(self):
        return self._vhi.__get__(self._state)

    @vhi.setter
    def vhi(self, value):
        return self._vhi.__set__(self._state, value)

    @property
    def vlog(self):
        return self._vlog.__get__(self._state)

    @vlog.setter
    def vlog(self, value):
        return self._vlog.__set__(self._state, value)

if __name__ == "__main__":

    from glue.core import Data

    class TestState(object):

        layer = CallbackProperty()
        comp = CallbackProperty(1)
        lower = CallbackProperty()
        higher = CallbackProperty()
        log = CallbackProperty()
        scale = CallbackProperty()

    state = TestState()
    state.layer = Data(x=np.arange(10),
                       y=np.arange(10) / 3.,
                       z=np.arange(10) - 5)

    helper = StateAttributeLimitsHelper(state, 'layer', 'comp',
                                        'lower', 'higher',
                                        percentile='scale', vlog='log')

    helper.component_id = state.layer.id['x']
