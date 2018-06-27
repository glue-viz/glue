# Combo helpers independent of GUI framework - these operate on
# SelectionCallbackProperty objects.

from __future__ import absolute_import, division, print_function

import weakref

from glue.core import Data, Subset
from glue.core.hub import HubListener
from glue.core.message import (DataReorderComponentMessage,
                               ComponentsChangedMessage,
                               DataCollectionAddMessage,
                               DataCollectionDeleteMessage,
                               DataUpdateMessage,
                               DataRenameComponentMessage)
from glue.external.echo import delay_callback, ChoiceSeparator
from glue.external.six import string_types

__all__ = ['ComponentIDComboHelper', 'ManualDataComboHelper',
           'DataCollectionComboHelper']


def unique_data_iter(datasets):
    """
    Return a list with only Data objects, with duplicates removed, but
    preserving the original order.
    """
    datasets_new = []
    for dataset in datasets:
        if isinstance(dataset, Data):
            if dataset not in datasets_new:
                datasets_new.append(dataset)
        else:
            if dataset.data not in datasets_new:
                datasets_new.append(dataset.data)
    return datasets_new


class ComboHelper(HubListener):
    """
    Base class for any combo helper represented by a SelectionCallbackProperty.

    This stores the state and selection property and exposes the ``state``,
    ``selection`` and ``choices`` properties.

    Parameters
    ----------
    state : :class:`~glue.core.state_objects.State`
        The state to which the selection property belongs
    selection_property : :class:`~glue.external.echo.core.SelectionCallbackProperty`
        The selection property representing the combo.
    """

    def __init__(self, state, selection_property):

        self._state = weakref.ref(state)
        self.selection_property = selection_property

    @property
    def state(self):
        """
        The state to which the selection property belongs.
        """
        return self._state()

    @property
    def selection(self):
        """
        The current selected value.
        """
        return getattr(self.state, self.selection_property)

    @selection.setter
    def selection(self, selection):
        return setattr(self.state, self.selection_property, selection)

    @property
    def choices(self):
        """
        The current valid choices for the combo.
        """
        prop = getattr(type(self.state), self.selection_property)
        return prop.get_choices(self.state)

    @choices.setter
    def choices(self, choices):
        with delay_callback(self.state, self.selection_property):
            prop = getattr(type(self.state), self.selection_property)
            prop.set_choices(self.state, choices)

    @property
    def display(self):
        """
        The current display function for the combo (the function that relates
        the Python objects to the display label)
        """
        prop = getattr(type(self.state), self.selection_property)
        return prop.get_display_func(self.state)

    @display.setter
    def display(self, display):
        prop = getattr(type(self.state), self.selection_property)
        return prop.set_display_func(self.state, display)

    def _on_rename(self, msg):
        # If a component ID is renamed, we don't need to refresh because the
        # list of actual component IDs is the same as before. However, we do
        # need to trigger a refresh of any GUI combos that use this, so we
        # make the property notify a change. However, if we are inside a
        # delay_callback block, the property will not be enabled, and notify()
        # won't have any effect, in which case we set the 'force_next_sync'
        # option which means that when exiting from the delay_callback block,
        # this property will show up as having changed
        prop = getattr(type(self.state), self.selection_property)
        if prop.enabled(self.state):
            prop.notify(self.state, self.selection, self.selection)
        else:
            prop.force_next_sync(self.state)


class ComponentIDComboHelper(ComboHelper):
    """
    The purpose of this class is to set up a combo (represented by a
    SelectionCallbackProperty) showing componentIDs for one or more datasets, and to
    update these componentIDs if needed, for example if new components are added
    to a dataset, or if componentIDs are renamed. This is a GUI
    framework-independent implementation.

    Parameters
    ----------
    state : :class:`~glue.core.state_objects.State`
        The state to which the selection property belongs
    selection_property : :class:`~glue.external.echo.core.SelectionCallbackProperty`
        The selection property representing the combo.
    data_collection : :class:`~glue.core.DataCollection`, optional
        The data collection to which the datasets belong - if specified,
        this is used to remove datasets from the combo when they are removed
        from the data collection.
    data : :class:`~glue.core.Data`, optional
        If specified, set up the combo for this dataset only and don't allow
        datasets to be added/removed
    numeric : bool, optional
        Show numeric components
    categorical : bool, optional
        Show categorical components
    pixel_coord : bool, optional
        Show pixel coordinate components
    world_coord : bool, optional
        Show world coordinate components
    derived : bool, optional
        Show derived components
    none : bool or str, optional
        Add an entry that means `None`. If a string, this is the display string
        that will be shown for the `None` entry, otherwise an empty string is
        shown.
    """

    def __init__(self, state, selection_property,
                 data_collection=None, data=None,
                 numeric=True, categorical=True,
                 pixel_coord=False, world_coord=False, derived=True, none=False):

        super(ComponentIDComboHelper, self).__init__(state, selection_property)

        if isinstance(none, string_types):
            self._none = True
            self._none_label = none
        else:
            self._none = none
            self._none_label = ''

        def display_func_label(cid):
            if cid is None:
                return self._none_label
            else:
                return cid.label

        self.display = display_func_label

        self._numeric = numeric
        self._categorical = categorical
        self._pixel_coord = pixel_coord
        self._world_coord = world_coord
        self._derived = derived

        if data is None:
            self._manual_data = False
            self._data = []
        else:
            self._manual_data = True
            self._data = [data]

        self._data_collection = data_collection
        if data_collection is None:
            self.hub = None
        else:
            if data_collection.hub is None:
                raise ValueError("Hub on data collection is not set")
            else:
                self.hub = data_collection.hub

        if data is not None:
            self.refresh()

    def clear(self):
        self._data.clear()
        self.refresh()

    @property
    def numeric(self):
        return self._numeric

    @numeric.setter
    def numeric(self, value):
        self._numeric = value
        self.refresh()

    @property
    def categorical(self):
        return self._categorical

    @categorical.setter
    def categorical(self, value):
        self._categorical = value
        self.refresh()

    @property
    def pixel_coord(self):
        return self._pixel_coord

    @pixel_coord.setter
    def pixel_coord(self, value):
        self._pixel_coord = value
        self.refresh()

    @property
    def world_coord(self):
        return self._world_coord

    @world_coord.setter
    def world_coord(self, value):
        self._world_coord = value
        self.refresh()

    @property
    def derived(self):
        return self._derived

    @derived.setter
    def derived(self, value):
        self._derived = value
        self.refresh()

    @property
    def none(self):
        return self._none

    @none.setter
    def none(self, value):
        if isinstance(value, string_types):
            self._none = True
            self._none_label = value
        else:
            self._none = value
            self._none_label = ''
        self.refresh()

    def append_data(self, data, refresh=True):

        if self._manual_data:
            raise Exception("Cannot change data in ComponentIDComboHelper "
                            "initialized from a single dataset")

        if isinstance(data, Subset):
            data = data.data

        if self.hub is None:
            if data.hub is not None:
                self.hub = data.hub
        elif data.hub is not self.hub:
            raise ValueError("Data Hub is different from current hub")

        if data not in self._data:
            self._data.append(data)
            if refresh:
                self.refresh()

    def remove_data(self, data):

        if self._manual_data:
            raise Exception("Cannot change data in ComponentIDComboHelper "
                            "initialized from a single dataset")

        if data in self._data:
            self._data.remove(data)
            self.refresh()

    def set_multiple_data(self, datasets):
        """
        Add multiple datasets to the combo in one go (and clear any previous datasets).

        Parameters
        ----------
        datasets : list
            The list of :class:`~glue.core.data.Data` objects to add
        """

        if self._manual_data:
            raise Exception("Cannot change data in ComponentIDComboHelper "
                            "initialized from a single dataset")

        try:
            self._data.clear()
        except AttributeError:  # PY2
            self._data[:] = []
        for data in unique_data_iter(datasets):
            self.append_data(data, refresh=False)
        self.refresh()

    @property
    def hub(self):
        return self._hub

    @hub.setter
    def hub(self, value):
        self._hub = value
        if value is not None:
            self.register_to_hub(value)

    def refresh(self, *args):

        choices = []

        if self._none:
            choices.append(None)

        for data in self._data:

            derived_components = [cid for cid in data.derived_components if cid.parent is data]

            if len(self._data) > 1:
                if data.label is None or data.label == '':
                    choices.append(ChoiceSeparator('Untitled Data'))
                else:
                    choices.append(ChoiceSeparator(data.label))

            cids = [ChoiceSeparator('Main components')]
            for cid in data.primary_components:
                if cid not in data.coordinate_components:
                    comp = data.get_component(cid)
                    if ((comp.numeric and self.numeric) or
                            (comp.categorical and self.categorical)):
                        cids.append(cid)
            if len(cids) > 1:
                if self.pixel_coord or self.world_coord or (self.derived and len(derived_components) > 0):
                    choices += cids
                else:
                    choices += cids[1:]

            if self.numeric and self.derived:
                cids = [ChoiceSeparator('Derived components')]
                for cid in derived_components:
                    cids.append(cid)
                if len(cids) > 1:
                    choices += cids

            if self.pixel_coord or self.world_coord:
                cids = [ChoiceSeparator('Coordinate components')]
                if self.pixel_coord:
                    cids += data.pixel_component_ids
                if self.world_coord:
                    cids += data.world_component_ids
                if len(cids) > 1:
                    choices += cids

        self.choices = choices

    def _filter_msg(self, msg):
        return msg.sender in self._data

    def register_to_hub(self, hub):
        hub.subscribe(self, DataRenameComponentMessage,
                      handler=self._on_rename,
                      filter=self._filter_msg)
        hub.subscribe(self, DataReorderComponentMessage,
                      handler=self.refresh,
                      filter=self._filter_msg)
        hub.subscribe(self, ComponentsChangedMessage,
                      handler=self.refresh,
                      filter=self._filter_msg)
        if self._data_collection is not None:
            hub.subscribe(self, DataCollectionDeleteMessage,
                          handler=self._remove_data)

    def _remove_data(self, msg):
        self.remove_data(msg.data)

    def unregister(self, hub):
        hub.unsubscribe_all(self)


class BaseDataComboHelper(ComboHelper):
    """
    This is a base class for helpers for combo boxes that need to show a list
    of data objects.

    Parameters
    ----------
    state : :class:`~glue.core.state_objects.State`
        The state to which the selection property belongs
    selection_property : :class:`~glue.external.echo.core.SelectionCallbackProperty`
        The selection property representing the combo.
    data_collection : :class:`~glue.core.DataCollection`
        The data collection to which the datasets belong - this is needed
        because if a dataset is removed from the data collection, we want to
        remove it here.
    """

    def __init__(self, state, selection_property, data_collection=None):

        super(BaseDataComboHelper, self).__init__(state, selection_property)

        def display_func_label(cid):
            return cid.label

        self.display = display_func_label

        self._component_id_helpers = []
        self.state.add_callback(self.selection_property, self.refresh_component_ids)

        self._data_collection = data_collection

        if data_collection is not None:
            if data_collection.hub is None:
                raise ValueError("Hub on data collection is not set")
            else:
                self.hub = data_collection.hub
        else:
            self.hub = None

    def refresh(self, *args):
        self.choices = [data for data in self._datasets]
        self.refresh_component_ids()

    def refresh_component_ids(self, *args):
        data = getattr(self.state, self.selection_property)
        for helper in self._component_id_helpers:
            helper.clear()
            if data is not None:
                helper.append_data(data)
            helper.refresh()

    def add_component_id_combo(self, combo):
        helper = ComponentIDComboHelper(combo)
        self._component_id_helpers.append_data(helper)
        if self._data is not None:
            helper.append_data(self._data)

    @property
    def hub(self):
        return self._hub

    @hub.setter
    def hub(self, value):
        self._hub = value
        if value is not None:
            self.register_to_hub(value)

    def register_to_hub(self, hub):
        pass

    def _on_data_update(self, msg):
        if msg.attribute == 'label':
            self._on_rename(msg)
        else:
            self.refresh()


class ManualDataComboHelper(BaseDataComboHelper):
    """
    This is a helper for combo boxes that need to show a list of data objects
    that is manually curated.

    Datasets are added and removed using the
    :meth:`~ManualDataComboHelper.append_data` and
    :meth:`~ManualDataComboHelper.remove_data` methods.

    Parameters
    ----------
    state : :class:`~glue.core.state_objects.State`
        The state to which the selection property belongs
    selection_property : :class:`~glue.external.echo.core.SelectionCallbackProperty`
        The selection property representing the combo.
    data_collection : :class:`~glue.core.DataCollection`
        The data collection to which the datasets belong - this is needed
        because if a dataset is removed from the data collection, we want to
        remove it here.
    """

    def __init__(self, state, selection_property, data_collection=None):

        super(ManualDataComboHelper, self).__init__(state, selection_property,
                                                    data_collection=data_collection)

        self._datasets = []

    def set_multiple_data(self, datasets):
        """
        Add multiple datasets to the combo in one go (and clear any previous datasets).

        Parameters
        ----------
        datasets : list
            The list of :class:`~glue.core.data.Data` objects to add
        """

        try:
            self._datasets.clear()
        except AttributeError:  # PY2
            self._datasets[:] = []
        for data in unique_data_iter(datasets):
            self.append_data(data, refresh=False)
        self.refresh()

    def append_data(self, data, refresh=True):
        if data in self._datasets:
            return
        if self.hub is None and data.hub is not None:
            self.hub = data.hub
        self._datasets.append(data)
        if refresh:
            self.refresh()

    def remove_data(self, data):
        if data not in self._datasets:
            return
        self._datasets.remove(data)
        self.refresh()

    def _remove_data_msg(self, msg):
        self.remove_data(msg.data)

    def _filter_msg(self, msg):
        return msg.sender in self._datasets

    def _filter_msg_dc(self, msg):
        return msg.sender is self._data_collection

    def register_to_hub(self, hub):

        super(ManualDataComboHelper, self).register_to_hub(hub)

        hub.subscribe(self, DataUpdateMessage,
                      handler=self._on_data_update,
                      filter=self._filter_msg)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=self._remove_data_msg,
                      filter=self._filter_msg_dc)


class DataCollectionComboHelper(BaseDataComboHelper):
    """
    This is a helper for combo boxes that need to show a list of data objects
    that is always in sync with a :class:`~glue.core.DataCollection`.

    Parameters
    ----------
    state : :class:`~glue.core.state_objects.State`
        The state to which the selection property belongs
    selection_property : :class:`~glue.external.echo.core.SelectionCallbackProperty`
        The selection property representing the combo.
    data_collection : :class:`~glue.core.DataCollection`
        The data collection with which to stay in sync
    """

    def __init__(self, state, selection_property, data_collection):

        super(DataCollectionComboHelper, self).__init__(state, selection_property,
                                                        data_collection=data_collection)

        self._datasets = data_collection

        self.refresh()

    def _filter_msg_in(self, msg):
        return msg.sender in self._datasets

    def _filter_msg_is(self, msg):
        return msg.sender is self._datasets

    def register_to_hub(self, hub):
        super(DataCollectionComboHelper, self).register_to_hub(hub)
        hub.subscribe(self, DataUpdateMessage,
                      handler=self._on_data_update,
                      filter=self._filter_msg_in)
        hub.subscribe(self, DataCollectionAddMessage,
                      handler=self.refresh,
                      filter=self._filter_msg_is)
        hub.subscribe(self, DataCollectionDeleteMessage,
                      handler=self.refresh,
                      filter=self._filter_msg_is)
