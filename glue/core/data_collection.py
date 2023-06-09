from contextlib import contextmanager

from glue.core.message import (DataCollectionAddMessage,
                               DataCollectionDeleteMessage,
                               ComponentsChangedMessage)
from glue.core.registry import Registry
from glue.core.link_manager import LinkManager
from glue.core.data import Data, BaseCartesianData
from glue.core.hub import Hub, HubListener
from glue.core.coordinates import WCSCoordinates
from glue.config import settings, data_translator
from glue.utils import as_list, common_prefix


__all__ = ['DataCollection']


class DataCollection(HubListener):

    """
    The top-level object for interacting with datasets in Glue.

    DataCollections have the following responsibilities:

        * Providing a way to retrieve and store data
        * Broadcasting messages when data are added or removed
        * Keeping each managed data set's list of
          :class:`~glue.core.component.DerivedComponent` instances up-to-date
        * Creating the hub that all other objects should use to communicate
          with one another (stored in ``self.hub``)

    Parameters
    ----------
    data : :class:`~glue.core.data.Data`, or `list` of such, optional
        The data objects to be stored in the collection.
    """

    def __init__(self, data=None):
        super(DataCollection, self).__init__()

        self._link_manager = LinkManager(self)
        self._data = []

        self.hub = None

        self._disable_sync_link_manager = 0
        self._subset_groups = []
        self.register_to_hub(Hub())
        self.extend(as_list(data or []))
        self._sg_count = 0

        self._link_manager.register_to_hub(self.hub)

    @property
    def data(self):
        """The :class:`~glue.core.data.Data` objects in the collection"""
        return self._data

    def append(self, data):
        """
        Add a new dataset to this collection.

        Appending emits a DataCollectionAddMessage.
        It also updates the list of DerivedComponents that each
        data set can work with.

        Parameters
        ----------
        data : :class:`~glue.core.data.BaseCartesianData`, or `list` of such
            The dataset to add.
        """

        if isinstance(data, list):
            self.extend(data)
            return

        if data in self:
            return

        if not isinstance(data, BaseCartesianData):
            raise TypeError("Only BaseCartesianData subclasses can be used at this time")

        self._data.append(data)

        if self.hub:
            data.register_to_hub(self.hub)
            for s in data.subsets:
                s.register()
            msg = DataCollectionAddMessage(self, data)
            self.hub.broadcast(msg)

        self._sync_link_manager()

    def extend(self, data):
        """
        Add several new datasets to this collection.

        See :meth:`append` for more information.

        Parameters
        ----------
        data : `iterable` of :class:`~glue.core.data.BaseCartesianData`
            The datasets to add.
        """
        # Wait until all datasets are added to sync the link manager
        with self._ignore_link_manager_update():
            for d in data:
                self.append(d)
        self._sync_link_manager()

    def remove(self, data):
        """
        Remove a data set from the collection, if present.

        Emits a DataCollectionDeleteMessage.

        data : :class:`~glue.core.data.Data`
            The data object to remove.
        """
        if data not in self._data:
            return
        self._data.remove(data)
        Registry().unregister(data, Data)
        if self.hub:
            msg = DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)

    def clear(self):
        with self._ignore_link_manager_update():
            for data in list(self):
                self.remove(data)

    def _sync_link_manager(self):
        """
        Update the LinkManager, so all the DerivedComponents
        for each data set are up-to-date.
        """

        if getattr(self, '_disable_sync_link_manager', False):
            return

        # Avoid circular calls
        with self._ignore_link_manager_update():
            self._link_manager.update_externally_derivable_components()

    @contextmanager
    def _ignore_link_manager_update(self):
        self._disable_sync_link_manager += 1
        yield
        self._disable_sync_link_manager -= 1

    @contextmanager
    def delay_link_manager_update(self):
        """
        Context manager to delay any updates to the link manager until the
        context is exited.

        This can be useful for improving performance if e.g. several datasets
        or links are being added to the data collection, since otherwise the
        link manager updates its internal tree representation of the links
        after each operation.
        """
        self._disable_sync_link_manager += 1
        yield
        self._disable_sync_link_manager -= 1
        self._sync_link_manager()

    @property
    def links(self):
        """
        Tuple of :class:`~glue.core.component_link.ComponentLink` objects.
        """
        return tuple(self._link_manager.links)

    @property
    def external_links(self):
        """
        Tuple of :class:`~glue.core.component_link.ComponentLink` objects.
        """
        return tuple(self._link_manager.external_links)

    def add_link(self, links):
        """
        Add one or more links to the data collection.

        This will auto-update the components in each data set.

        Parameters
        ----------
        links : :class:`~glue.core.component_link.ComponentLink`, or `iterable` of such, or :class:`~glue.core.link_helpers.LinkCollection`
            The links to add.
        """
        self._link_manager.add_link(links, update_external=not self._disable_sync_link_manager)

    def remove_link(self, links):
        """
        Remove one or more links from the data collection.

        This will auto-update the components in each data set.

        Parameters
        ----------
        links : :class:`~glue.core.component_link.ComponentLink`, or `iterable` of such, or :class:`~glue.core.link_helpers.LinkCollection`
            The links to remove.
        """
        self._link_manager.remove_link(links, update_external=not self._disable_sync_link_manager)

    def _merge_link(self, link):
        pass

    def set_links(self, links):
        """
        Override the links in the collection, and update data objects as
        necessary.

        Parameters
        ----------
        links : :class:`~glue.core.component_link.ComponentLink`, or `iterable` of such, or :class:`~glue.core.link_helpers.LinkCollection`
            The new links.
        """
        self._link_manager.clear_links()
        self._link_manager.add_link(links, update_external=not self._disable_sync_link_manager)

    def register_to_hub(self, hub):
        """ Register managed data objects to a hub.

        Parameters
        ----------
        hub : :class:`~glue.core.hub.Hub`
            The hub to register with.
        """
        if self.hub is hub:
            return
        if self.hub is not None:
            raise RuntimeError("Data Collection already registered "
                               "to a different Hub")

        if not isinstance(hub, Hub):
            raise TypeError("Input is not a Hub object: %s" % type(hub))
        self.hub = hub

        # re-assign all data, subset hub instances to this hub
        for d in self._data:
            d.register_to_hub(hub)
            for s in d.subsets:
                s.register()

        hub.subscribe(self, ComponentsChangedMessage,
                      lambda msg: self._sync_link_manager(),
                      filter=lambda x: x.sender in self._data)

    def new_subset_group(self, label=None, subset_state=None, **kwargs):
        """
        Create and return a new Subset Group.

        Parameters
        ----------
        label : `str`
            The label to assign to the group.
        subset_state : :class:`~glue.core.subset.SubsetState`
            The state to initialize the group with.

        Returns
        -------
        :class:`~glue.core.subset_group.SubsetGroup`
        """
        from glue.core.subset_group import SubsetGroup
        kwargs.setdefault("color", settings.SUBSET_COLORS[self._sg_count % len(settings.SUBSET_COLORS)])
        self._sg_count += 1
        label = label or 'Subset %i' % self._sg_count

        result = SubsetGroup(label=label, subset_state=subset_state, **kwargs)
        self._subset_groups.append(result)
        result.register(self)
        return result

    def remove_subset_group(self, subset_grp):
        """
        Remove an existing :class:`~glue.core.subset_group.SubsetGroup`
        """
        if subset_grp not in self._subset_groups:
            return

        # remove from list first, so that group appears deleted
        # by the time the first SubsetDelete message is broadcast
        self._subset_groups.remove(subset_grp)
        for s in subset_grp.subsets:
            s.delete()
        subset_grp.unregister(self.hub)

    def suggest_merge_label(self, *data):
        """
        Determine what merge label to suggest given datasets
        """

        # Find longest common prefix for data
        suggestion = common_prefix([d.label for d in data])

        if len(suggestion) < 3:
            suggestion = 'Merged data'

        # Now check if the suggestion already exists, and if so add a suffix
        labels = self.labels
        if suggestion in labels:
            suffix = 2
            while "{0} [{1}]".format(suggestion, suffix) in labels:
                suffix += 1
            suggestion = "{0} [{1}]".format(suggestion, suffix)

        return suggestion

    @property
    def labels(self):
        return [d.label for d in self]

    def merge(self, *data, **kwargs):
        """
        Merge two or more datasets into a single dataset.

        This has the following effects:

        All components from all datasets are added to the first argument.
        All datasets except the first argument are removed from the collection.
        Any component name conflicts are disambiguated.
        The pixel and world components apart from the first argument are discarded.

        Parameters
        ----------
        data : `iterable` of :class:`~glue.core.data.Data`
            Two or more datasets to be added to this collection.

        Notes
        -----
        All arguments must have the same shape.

        Returns
        -------
        self
        """
        if len(data) < 2:
            raise ValueError("merge requires 2 or more arguments")
        shp = data[0].shape
        for d in data:
            if d.shape != shp:
                raise ValueError("All arguments must have the same shape")

        label = kwargs.get('label', data[0].label)

        master = Data(label=label)
        self.append(master)

        master.coords = data[0].coords
        for i, d in enumerate(data):
            if isinstance(d.coords, WCSCoordinates):
                master.coords = d.coords
                break

        # Find ambiguous components (ones which have labels in more than one
        # dataset

        from collections import Counter
        clabel_count = Counter([c.label for d in data for c in d.main_components + d.derived_components])

        for d in data:

            for c in d.components:

                if c in master.components:  # already present (via a link)
                    continue

                # Don't include coordinate components here as they will be
                # recomputed separately once the first non-coordinate component
                # is added.
                if c in d.coordinate_components:
                    continue

                lbl = c.label

                if clabel_count[lbl] > 1:
                    lbl = lbl + " [{0}]".format(d.label)

                c._label = lbl
                c.parent = master
                master.add_component(d.get_component(c), c)

            self.remove(d)

        return master

    @property
    def subset_groups(self):
        """
        `tuple` of current :class:`Subset Groups <glue.core.subset_group.SubsetGroup>`
        """
        return tuple(self._subset_groups)

    def __contains__(self, obj):
        return (obj in self._data or
                    obj in self.subset_groups or
                    any([data.label == obj for data in self._data]))

    def __getitem__(self, key):
        if isinstance(key, str):
            matches = [data for data in self._data if data.label == key]
            if len(matches) == 0:
                raise ValueError("No data found with the label '{0}'".format(key))
            elif len(matches) > 1:
                raise ValueError("Several datasets were found with the label '{0}'".format(key))
            else:
                return matches[0]
        else:
            return self._data[key]

    def __setitem__(self, key, data):
        """
        Add a dataset to the data collection.

        This can be either a :class:`~glue.core.data.Data` object, which will
        then have its label set to the specified key, or another kind of
        object which will be automatically translated into a
        :class:`~glue.core.data.Data` object.
        """

        if not isinstance(key, str):
            raise TypeError("item key should be a string, but got {0}".format(type(key)))

        if not isinstance(data, BaseCartesianData):

            handler, preferred = data_translator.get_handler_for(data)

            data = handler.to_data(data)
            data._preferred_translation = preferred

        data.label = key

        for existing_data in self._data[:]:
            if existing_data.label == key:
                self.remove(existing_data)

        self.append(data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def index(self, item):
        return self._data.index(item)

    def __str__(self):
        if len(self) == 1:
            result = "DataCollection (1 data set)\n\t"
        else:
            result = "DataCollection (%i data sets)\n\t" % len(self)

        result += '\n\t'.join("%3i: %s" % (i, d.label) for
                              i, d in enumerate(self))
        return result

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return True

    def __nonzero__(self):
        return True
