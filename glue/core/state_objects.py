from __future__ import absolute_import, division, print_function

from collections import defaultdict

import numpy as np

from glue.core import Subset
from glue.external.echo import (delay_callback, CallbackProperty,
                                HasCallbackProperties, CallbackList)
from glue.core.state import saver, loader
from glue.core.component_id import PixelComponentID

__all__ = ['State', 'StateAttributeCacheHelper',
           'StateAttributeLimitsHelper', 'StateAttributeSingleValueHelper']


@saver(CallbackList)
def _save_callback_list(items, context):
    return {'values': [context.id(item) for item in items]}


@loader(CallbackList)
def _load_callback_list(rec, context):
    return [context.object(obj) for obj in rec['values']]


class State(HasCallbackProperties):
    """
    A class to represent the state of a UI element. Initially this doesn't add
    anything compared to HasCallbackProperties, but functionality will be added
    over time.
    """

    def __init__(self, **kwargs):
        super(State, self).__init__()
        self.update_from_dict(kwargs)

    def update_from_state(self, state):
        """
        Update this state using the values from another state.

        Parameters
        ----------
        state : `~glue.core.state_objects.State`
            The state to use the values from
        """
        self.update_from_dict(state.as_dict())

    def update_from_dict(self, properties):
        """
        Update this state using the values from a dictionary of attributes.

        Parameters
        ----------
        properties : dict
            The dictionary containing attribute/value pairs.
        """

        if len(properties) == 0:
            return

        # Group properties into priority so as to be able to update them in
        # chunks and not fire off callback events between every single one.
        groups = defaultdict(list)
        for name in properties:
            if self.is_callback_property(name):
                groups[self._update_priority(name)].append(name)

        for priority in sorted(groups, reverse=True):
            with delay_callback(self, *groups[priority]):
                for name in groups[priority]:
                    setattr(self, name, properties[name])

    def as_dict(self):
        """
        Return the current state as a dictionary of attribute/value pairs.
        """
        properties = {}
        for name in dir(self):
            if not name.startswith('_') and self.is_callback_property(name):
                properties[name] = getattr(self, name)
        return properties

    def __gluestate__(self, context):
        return {'values': dict((key, context.id(value)) for key, value in self.as_dict().items())}

    def _update_priority(self, name):
        return 0

    @classmethod
    def __setgluestate__(cls, rec, context):
        properties = dict((key, context.object(value)) for key, value in rec['values'].items())
        result = cls(**properties)
        return result


class StateAttributeCacheHelper(object):
    """
    Generic class to help with caching values on a per-attribute basis

    Parameters
    ----------
    state : :class:`glue.core.state_objects.State`
        The state object with the callback properties to cache
    attribute : str
        The attribute name - this will be populated once a dataset is assigned
        to the helper
    cache : dict, optional
        A dictionary that can be used to hold the cache. This option can be used
        if a common cache should be shared between different helpers.
    kwargs
        Additional keyword arguments are taken to be values that should be
        used/cached. The key should be the name to be understood by sub-classes
        of this base class, and the value should be the name of the attribute
        in the state.
    """

    def __init__(self, state, attribute, cache=None, **kwargs):

        self._state = state
        self._attribute = attribute
        self._values = dict((key, kwargs[key]) for key in self.values_names if key in kwargs)
        self._modifiers = dict((key, kwargs[key]) for key in self.modifiers_names if key in kwargs)

        self._attribute_lookup = {'attribute': self._attribute}
        self._attribute_lookup.update(self._values)
        self._attribute_lookup.update(self._modifiers)
        self._attribute_lookup_inv = {v: k for k, v in self._attribute_lookup.items()}

        self._state.add_callback(self._attribute, self._update_attribute)

        self._state.add_global_callback(self._update_values)

        # NOTE: don't use self._cache = cache or {} here since if the initial
        #       cache is empty it will evaluate as False!
        if cache is None:
            self._cache = {}
        else:
            self._cache = cache

    @property
    def data_values(self):
        # For subsets in 'data' mode, we want to compute the limits based on
        # the full dataset, not just the subset.
        if isinstance(self.data, Subset):
            return self.data.data[self.component_id]
        else:
            return self.data[self.component_id]

    @property
    def data_component(self):
        # For subsets in 'data' mode, we want to compute the limits based on
        # the full dataset, not just the subset.
        if isinstance(self.data, Subset):
            return self.data.data.get_component(self.component_id)
        else:
            return self.data.get_component(self.component_id)

    def invalidate_cache(self):
        self._cache.clear()

    @property
    def data(self):
        if self.attribute is None:
            return None
        else:
            return self.attribute.parent

    @property
    def component_id(self):
        if self.attribute is None:
            return None
        else:
            return self.attribute

    def set_cache(self, cache):
        self._cache = cache
        self._update_attribute()

    def _update_attribute(self, *args):
        if self.component_id in self._cache:
            # The component ID already exists in the cache, so just revert to
            # that version of the values/settings.
            self.set(cache=False, **self._cache[self.component_id])
        else:
            # We need to compute the values for the first time
            self.update_values(attribute=self.component_id, use_default_modifiers=True)

    def _update_values(self, **properties):

        if hasattr(self, '_in_set'):
            if self._in_set:
                return
        if self.attribute is None:
            return
        properties = dict((self._attribute_lookup_inv[key], value)
                          for key, value in properties.items() if key in self._attribute_lookup_inv and self._attribute_lookup_inv[key] != 'attribute')
        if len(properties) > 0:
            self.update_values(**properties)

    def _modifiers_as_dict(self):
        return dict((prop, getattr(self, prop)) for prop in self.modifiers_names if prop in self._modifiers)

    def _values_as_dict(self):
        return dict((prop, getattr(self, prop)) for prop in self.values_names if prop in self._values)

    def _update_cache(self):
        if self.component_id is not None:
            self._cache[self.component_id] = {}
            self._cache[self.component_id].update(self._modifiers_as_dict())
            self._cache[self.component_id].update(self._values_as_dict())

    def __getattr__(self, attribute):
        if attribute in self._attribute_lookup:
            return getattr(self._state, self._attribute_lookup[attribute])
        else:
            raise AttributeError(attribute)

    def __setattr__(self, attribute, value):
        if attribute.startswith('_') or attribute not in self._attribute_lookup:
            return object.__setattr__(self, attribute, value)
        else:
            return setattr(self._state, self._attribute_lookup[attribute], value)

    def set(self, cache=True, **kwargs):

        self._in_set = True

        extra_kwargs = set(kwargs.keys()) - set(self.values_names) - set(self.modifiers_names)

        if len(extra_kwargs) > 0:
            raise ValueError("Invalid properties: {0}".format(extra_kwargs))

        with delay_callback(self._state, *self._attribute_lookup.values()):
            for prop, value in kwargs.items():
                setattr(self, prop, value)

        if cache:
            self._update_cache()

        self._in_set = False


class StateAttributeLimitsHelper(StateAttributeCacheHelper):
    """
    This class is a helper for attribute-dependent min/max level values. It
    is equivalent to AttributeLimitsHelper but operates on State objects and
    is GUI-independent.

    Parameters
    ----------
    attribute : str
        The attribute name - this will be populated once a dataset is assigned
        to the helper.
    percentile_subset : int
        How many points to use at most for the percentile calculation (using all
        values is highly inefficient and not needed)
    margin : float
        Whether to add a margin to the range of values determined. This should be
        given as a floating point value that will be multiplied by the range of
        values to give the margin to add to the limits.
    lower, upper : str
        The fields for the lower/upper levels
    percentile : ``QComboBox`` instance, optional
        The scale mode combo - this will be populated by presets such as
        Min/Max, various percentile levels, and Custom.
    log : bool
        Whether the limits are in log mode (in which case only positive values
        are used when finding the limits)

    Notes
    -----

    Once the helper is instantiated, the data associated with the helper can be
    set/changed with:

    >>> helper = AttributeLimitsHelper(...)
    >>> helper.data = data

    The data can also be passed to the initializer as described in the list of
    parameters above.
    """

    values_names = ('lower', 'upper')
    modifiers_names = ('log', 'percentile')

    def __init__(self, state, attribute, percentile_subset=10000, margin=0, cache=None, **kwargs):

        super(StateAttributeLimitsHelper, self).__init__(state, attribute, cache=cache, **kwargs)

        self.margin = margin
        self.percentile_subset = percentile_subset
        self.subset_indices = None

        if self.attribute is not None:

            if (self.lower is not None and self.upper is not None and getattr(self, 'percentile', None) is None):
                # If the lower and upper limits are already set, we need to make
                # sure we don't override them, so we set the percentile mode to
                # custom if it isn't already set.
                self.set(percentile='Custom')
            else:
                # Otherwise, we force the recalculation or the fetching from
                # cache of the limits based on the current attribute
                self._update_attribute()

    def update_values(self, force=False, use_default_modifiers=False, **properties):

        if not force and not any(prop in properties for prop in ('attribute', 'percentile', 'log')):
            self.set(percentile='Custom')
            return

        if use_default_modifiers:
            percentile = 100
            log = False
        else:
            percentile = getattr(self, 'percentile', None) or 100
            log = getattr(self, 'log', None) or False

        if not force and (percentile == 'Custom' or not hasattr(self, 'data') or self.data is None):

            self.set(percentile=percentile, log=log)

        else:

            # Shortcut if the component ID is a pixel component ID
            if isinstance(self.component_id, PixelComponentID) and percentile == 100 and not log:
                lower = -0.5
                upper = self.data.shape[self.component_id.axis] - 0.5
                self.set(lower=lower, upper=upper, percentile=percentile, log=log)
                return

            exclude = (100 - percentile) / 2.

            data_values = self.data_values

            if data_values.size > self.percentile_subset:
                if self.subset_indices is None or self.subset_indices[0] != data_values.size:
                    self.subset_indices = (data_values.size,
                                           np.random.randint(0, data_values.size,
                                                             self.percentile_subset))
                data_values = data_values.ravel()[self.subset_indices[1]]

            if log:
                data_values = data_values[data_values > 0]
                if len(data_values) == 0:
                    self.set(lower=0.1, upper=1, percentile=percentile, log=log)
                    return

            # NOTE: we can't use np.nanmin/np.nanmax or nanpercentile below as
            # they don't exclude inf/-inf
            if data_values.dtype.kind != 'M':
                data_values = data_values[np.isfinite(data_values)]

            if data_values.size > 0:

                if percentile == 100:

                    if data_values.dtype.kind == 'M':
                        lower = data_values.min()
                        upper = data_values.max()
                    else:
                        lower = np.min(data_values)
                        upper = np.max(data_values)

                else:

                    lower = np.percentile(data_values, exclude)
                    upper = np.percentile(data_values, 100 - exclude)

                if self.data_component.categorical:
                    lower = np.floor(lower - 0.5) + 0.5
                    upper = np.ceil(upper + 0.5) - 0.5

                if log:
                    value_range = np.log10(upper / lower)
                    lower /= 10.**(value_range * self.margin)
                    upper *= 10.**(value_range * self.margin)
                else:
                    value_range = upper - lower
                    lower -= value_range * self.margin
                    upper += value_range * self.margin

            else:

                lower = 0.
                upper = 1.

            self.set(lower=lower, upper=upper, percentile=percentile, log=log)

    def flip_limits(self):
        self.set(lower=self.upper, upper=self.lower)


class StateAttributeSingleValueHelper(StateAttributeCacheHelper):

    values_names = ('value',)
    modifiers_names = ()

    def __init__(self, state, attribute, function, mode='values', **kwargs):
        self._function = function
        super(StateAttributeSingleValueHelper, self).__init__(state, attribute, **kwargs)
        if self.attribute is not None:
            self._update_attribute()
        if mode in ('values', 'component'):
            self.mode = mode
        else:
            raise ValueError('mode should be one of "values" or "component"')

    def update_values(self, use_default_modifiers=False, **properties):
        if not any(prop in properties for prop in ('attribute',)) or self.data is None:
            self.set()
        else:
            if self.mode == 'values':
                arg = self.data_values
            else:
                arg = self.data_component
            self.set(value=self._function(arg))


class StateAttributeHistogramHelper(StateAttributeCacheHelper):

    values_names = ('lower', 'upper', 'n_bin')
    modifiers_names = ()

    def __init__(self, *args, **kwargs):

        self._max_n_bin = kwargs.pop('max_n_bin', 30)
        self._default_n_bin = kwargs.pop('default_n_bin', 15)

        common_n_bin_att = kwargs.pop('common_n_bin', None)

        super(StateAttributeHistogramHelper, self).__init__(*args, **kwargs)

        if common_n_bin_att is not None:
            if getattr(self._state, common_n_bin_att):
                self._common_n_bin = self._default_n_bin
            else:
                self._common_n_bin = None
            self._state.add_callback(common_n_bin_att, self._update_common_n_bin)
        else:
            self._common_n_bin = None

    def _apply_common_n_bin(self):
        for att in self._cache:
            cmp = self.data.get_component(att)
            if not cmp.categorical:
                self._cache[att]['n_bin'] = self._common_n_bin

    def _update_common_n_bin(self, common_n_bin):
        if common_n_bin:
            if self.data_component.categorical:
                self._common_n_bin = self._default_n_bin
            else:
                self._common_n_bin = self.n_bin
            self._apply_common_n_bin()
        else:
            self._common_n_bin = None

    def update_values(self, force=False, use_default_modifiers=False, **properties):

        if not force and not any(prop in properties for prop in ('attribute', 'n_bin')) or self.data is None:
            self.set()
            return

        comp = self.data_component

        if 'n_bin' in properties:
            self.set()
            if self._common_n_bin is not None and not comp.categorical:
                self._common_n_bin = properties['n_bin']
                self._apply_common_n_bin()

        if 'attribute' in properties or force:

            if comp.categorical:

                n_bin = max(1, min(comp.categories.size, self._max_n_bin))
                lower = -0.5
                upper = lower + comp.categories.size

            else:

                if self._common_n_bin is None:
                    n_bin = self._default_n_bin
                else:
                    n_bin = self._common_n_bin

                values = self.data_values

                # NOTE: we can't use np.nanmin/np.nanmax or nanpercentile below as
                # they don't exclude inf/-inf
                if values.dtype.kind != 'M':
                    values = values[np.isfinite(values)]

                if values.size > 0:
                    lower = values.min()
                    upper = values.max()
                else:
                    lower = 0.
                    upper = 1.

            self.set(lower=lower, upper=upper, n_bin=n_bin)


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
                                        percentile='scale', log='log')

    helper.component_id = state.layer.id['x']
