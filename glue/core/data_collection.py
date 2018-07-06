from __future__ import absolute_import, division, print_function

from contextlib import contextmanager

from glue.core.message import (DataCollectionAddMessage,
                               DataCollectionDeleteMessage,
                               ComponentsChangedMessage)
from glue.core.registry import Registry
from glue.core.link_manager import LinkManager
from glue.core.data import Data, BaseCartesianData
from glue.core.hub import Hub, HubListener
from glue.core.coordinates import WCSCoordinates
from glue.config import settings
from glue.utils import as_list, common_prefix


__all__ = ['DataCollection']


class DataCollection(HubListener):

    """The top-level object for interacting with datasets in Glue.

    DataCollections have the following responsibilities:

       * Providing a way to retrieve and store data
       * Broadcasting messages when data are added or removed
       * Keeping each managed data set's list of
         :class:`~glue.core.component.DerivedComponent` instances up-to-date
       * Creating the hub that all other objects should use to communicate
         with one another (stored in ``self.hub``)
    """

    def __init__(self, data=None):
        """
        :param data: :class:`~glue.core.data.Data` object, or list of such objects
        """
        super(DataCollection, self).__init__()

        self._link_manager = LinkManager(self)
        self._data = []

        self.hub = None

        self._subset_groups = []
        self.register_to_hub(Hub())
        self.extend(as_list(data or []))
        self._sg_count = 0

        self._link_manager.register_to_hub(self.hub)

    @property
    def data(self):
        """ The :class:`~glue.core.data.Data` objects in the collection """
        return self._data

    def append(self, data):
        """ Add a new dataset to this collection.

        Appending emits a DataCollectionAddMessage.
        It also updates the list of DerivedComponents that each
        data set can work with.

        :param data: :class:`~glue.core.data.BaseCartesianData` object to add
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
        """Add several new datasets to this collection

        See :meth:`append` for more information

        :param data: List of data objects to add
        """
        # Wait until all datasets are added to sync the link manager
        with self._no_sync_link_manager():
            for d in data:
                self.append(d)
        self._sync_link_manager()

    def remove(self, data):
        """ Remove a data set from the collection

        Emits a DataCollectionDeleteMessage

        :param data: the object to remove
        :type data: :class:`~glue.core.data.Data`
        """
        if data not in self._data:
            return
        self._data.remove(data)
        Registry().unregister(data, Data)
        if self.hub:
            msg = DataCollectionDeleteMessage(self, data)
            self.hub.broadcast(msg)

    def _sync_link_manager(self):
        """ update the LinkManager, so all the DerivedComponents
        for each data set are up-to-date
        """

        if getattr(self, '_disable_sync_link_manager', False):
            return

        # Avoid circular calls
        with self._no_sync_link_manager():
            self._link_manager.update_externally_derivable_components()

    @contextmanager
    def _no_sync_link_manager(self):
        self._disable_sync_link_manager = True
        yield
        self._disable_sync_link_manager = False

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
        """Add one or more links to the data collection.

        This will auto-update the components in each data set

        :param links:
           The links to add. A scalar or list of
           :class:`~glue.core.component_link.ComponentLink`
           instances, or a :class:`~glue.core.link_helpers.LinkCollection`
        """
        self._link_manager.add_link(links)

    def remove_link(self, links):
        """
        Remove one or more links from the data collection.

        This will auto-update the components in each data set

        :param links:
           The links to remove. A scalar or list of
           :class:`~glue.core.component_link.ComponentLink`
           instances, or a :class:`~glue.core.link_helpers.LinkCollection`
        """
        self._link_manager.remove_link(links)

    def _merge_link(self, link):
        pass

    def set_links(self, links):
        """
        Override the links in the collection, and update data objects as
        necessary.

        :param links: The new links. An iterable of
            :class:`~glue.core.component_link.ComponentLink` instances
        """
        self._link_manager.clear_links()
        self._link_manager.add_link(links)

    def register_to_hub(self, hub):
        """ Register managed data objects to a hub.

        :param hub: The hub to register with
        :type hub: :class:`~glue.core.hub.Hub`
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

    def new_subset_group(self, label=None, subset_state=None):
        """
        Create and return a new Subset Group.

        :param label: The label to assign to the group
        :type label: str
        :param subset_state: The state to initialize the group with
        :type subset_state: :class:`~glue.core.subset.SubsetState`

        :returns: A new :class:`~glue.core.subset_group.SubsetGroup`
        """
        from glue.core.subset_group import SubsetGroup
        color = settings.SUBSET_COLORS[self._sg_count % len(settings.SUBSET_COLORS)]
        self._sg_count += 1
        label = label or 'Subset %i' % self._sg_count

        result = SubsetGroup(color=color, label=label, subset_state=subset_state)
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

        All components from all datasets are added to the first argument
        All datasets except the first argument are removed from the collection
        Any component name conflicts are disambiguated
        The pixel and world components apart from the first argument are discarded

        :note: All arguments must have the same shape

        :param data: One or more :class:`~glue.core.data.Data` instances.
        :returns: self
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

        return self

    @property
    def subset_groups(self):
        """
        tuple of current :class:`Subset Groups <glue.core.subset_group.SubsetGroup>`
        """
        return tuple(self._subset_groups)

    def __contains__(self, obj):
        return obj in self._data or obj in self.subset_groups

    def __getitem__(self, key):
        return self._data[key]

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
