from __future__ import absolute_import, division, print_function

from glue.core.util import disambiguate
from glue.core.message import (DataCollectionAddMessage,
                               DataCollectionDeleteMessage,
                               DataAddComponentMessage)
from glue.core.registry import Registry
from glue.core.link_manager import LinkManager
from glue.core.data import Data
from glue.core.hub import Hub, HubListener
from glue.config import settings
from glue.utils import as_list


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
        self._link_manager = LinkManager()
        self._data = []

        self.hub = None

        self._subset_groups = []
        self.register_to_hub(Hub())
        self.extend(as_list(data or []))
        self._sg_count = 0

    @property
    def data(self):
        """ The :class:`~glue.core.data.Data` objects in the collection """
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

        See :meth:`append` for more information

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
        """
        Tuple of :class:`~glue.core.component_link.ComponentLink` objects.
        """
        return tuple(self._link_manager.links)

    def add_link(self, links):
        """Add one or more links to the data collection.

        This will auto-update the components in each data set

        :param links:
           The links to add. A scalar or list of
           :class:`~glue.core.component_link.ComponentLink`
           instances, or a :class:`~glue.core.link_helpers.LinkCollection`
        """
        self._link_manager.add_link(links)
        for d in self._data:
            self._link_manager.update_data_components(d)

    def _merge_link(self, link):
        pass

    def set_links(self, links):
        """Override the links in the collection, and update data
        objects as necessary.

        :param links: The new links. An iterable of
            :class:`~glue.core.component_link.ComponentLink` instances
        """
        self._link_manager.clear()
        for link in links:
            self._link_manager.add_link(link)

        for d in self._data:
            self._link_manager.update_data_components(d)

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

        hub.subscribe(self, DataAddComponentMessage,
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
        label = label or "%i" % (self._sg_count)

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

        for d in data:
            skip = d.pixel_component_ids + d.world_component_ids
            for c in d.components:
                if c in skip:
                    continue

                if c in master.components:  # already present (via a link)
                    continue

                taken = [_.label for _ in master.components]
                lbl = c.label

                # Special-case 'PRIMARY', rename to data label
                if lbl == 'PRIMARY':
                    lbl = d.label

                # First-pass disambiguation, try component_data
                if lbl in taken:
                    lbl = '%s_%s' % (lbl, d.label)

                lbl = disambiguate(lbl, taken)
                c._label = lbl
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
