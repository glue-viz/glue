from .hub import Hub, HubListener
from .data import Data
from .link_manager import LinkManager
from .registry import Registry
from .visual import COLORS
from .message import (DataCollectionAddMessage,
                      DataCollectionDeleteMessage,
                      DataAddComponentMessage)
from .util import as_list

__all__ = ['DataCollection']


class DataCollection(HubListener):

    """DataCollections manage sets of data. They have the following
    responsibilities:

       * Providing a way to retrieve and store data
       * Broadcasting messages when data are added or removed
       * Keeping each managed data set's list of DerivedComponents up-to-date
       * Creating the hub that all other objects should use to communicate
         with one another (stored in DataCollection.hub)
    """

    def __init__(self, data=None, hub=None):
        """
        :param data: glue.Data object, or list of such objects (optional)
                      These objects will be auto-appended to the collection
        """
        super(DataCollection, self).__init__()
        self._link_manager = LinkManager()
        self._data = []
        self._subset_groups = []
        self.hub = None
        self.register_to_hub(Hub())
        self.extend(as_list(data or []))
        self._sg_count = 0

    @property
    def data(self):
        """ The data objects in the collection (Read Only) """
        return self._data

    def append(self, data):
        """ Add a new dataset to this collection.

        Appending emits a DataCollectionAddMessage.
        It also updates the list of DerivedComponents that each
        data set can work with.

        :param data: :class:`~glue.core.data.Data` object to add
        """
        if isinstance(data, list):
            self.extend(data)
            return
        if data in self:
            return
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

        See :meth:`~DataCollection.append` for more information

        :param data: List of data objects to add
        """
        [self.append(d) for d in data]

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

        # add any links in the data
        for d in self._data:
            for derived in d.derived_components:
                self._link_manager.add_link(d.get_component(derived).link)
            for link in d.coordinate_links:
                self._link_manager.add_link(link)

        for d in self._data:
            self._link_manager.update_data_components(d)

    @property
    def links(self):
        return tuple(self._link_manager.links)

    def add_link(self, links):
        """Add one or more links to the data collection.
        This will auto-update the components in each data set

        :param links: The links to add
        :type links: A scalar or list of
                     :class:`~glue.core.component_link.ComponentLink`
                     instances,
                     or a :class:`~glue.core.link_helpers.LinkCollection`
        """
        self._link_manager.add_link(links)
        for d in self._data:
            self._link_manager.update_data_components(d)

    def _merge_link(self, link):
        pass

    def set_links(self, links):
        """Override the links in the collection, and update data
        objects as necessary

        :param links: The new links
        :type links: An iterable of
                     :class:`~glue.core.component_link.ComponentLInk`
                     instances
        """
        self._link_manager.clear()
        for link in links:
            self._link_manager.add_link(link)

        for d in self._data:
            self._link_manager.update_data_components(d)

    def register_to_hub(self, hub):
        """ Register managed data objects to a hub
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

        hub.subscribe(self, DataAddComponentMessage,
                      lambda msg: self._sync_link_manager(),
                      filter=lambda x: x.sender in self._data)

    def new_subset_group(self):
        """
        Create and return a new :class:`~glue.core.subset_group.SubsetGroup`
        """
        from .subset_group import SubsetGroup
        color = COLORS[self._sg_count % len(COLORS)]
        self._sg_count += 1
        label = "%i" % (self._sg_count)

        result = SubsetGroup(color=color, label=label)
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

    @property
    def subset_groups(self):
        """ View of current subset groups """
        return tuple(self._subset_groups)

    def __contains__(self, obj):
        return obj in self._data or obj in self.subset_groups

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __str__(self):
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
