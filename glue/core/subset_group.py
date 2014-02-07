"""
A :class:`~glue.core.subset_group.Subset Group` unites a group of
:class:`~glue.core.Subset` instances together with a consistent state,
label, and style.

While subsets are internally associated with particular datasets, it's
confusing for the user to juggle multiple similar or identical
subsets, applied to different datasets. Because of this, the GUI
manages SubsetGroups, and presents each group to the user as a single
entity. The individual subsets are held in-sync by the SubsetGroup.

Client code should *only* create Subset Groups via
DataCollection.new_subset_group. It should *not* call Data.add_subset
or Data.new_subset directly
"""
from . import Subset
from .subset import SubsetState
from .util import Pointer
from .hub import HubListener
from .message import (DataCollectionAddMessage,
                      DataCollectionDeleteMessage
                      )


class GroupedSubset(Subset):
    """
    A member of a SubsetGroup, whose internal representation
    is shared with other group members
    """
    subset_state = Pointer('group.subset_state')
    label = Pointer('group.label')

    def __init__(self, data, group, **kwargs):
        """
        :param data: :class:`~glue.core.data.Data` instance to bind to
        :param group: :class:`~glue.core.subset_group.SubsetGroup`
        """
        self.group = group
        self._style_override = None
        super(GroupedSubset, self).__init__(data, **kwargs)

    @property
    def style(self):
        return self._style_override or self.group.style

    @style.setter
    def style(self, value):
        self.group.style = value

    def override_style(self, attr, value):
        style = self.group.style.copy()
        style.parent=self
        setattr(style, attr, value)
        self._style_override = style
        self.broadcast('style')

    def clear_override_style(self):
        self._style_override = None
        self.broadcast('style')

    def __eq__(self, other):
        return other is self


class SubsetGroup(HubListener):
    def __init__(self, data):
        """
        Create a new SubsetGroup from a DataCollection.

        Note: By convention, SubsetGroups should be created via
        DataCollection.new_subset.

        :param data: :class:`~glue.core.DataCollection`
        """
        self.subsets = []
        self.state = SubsetState()
        self.label = ''
        self.style = None
        self.register_to_hub(data.hub)

        for d in data:
            s = GroupedSubset(d, self)
            d.add_subset(s)
            self.subsets.append(s)

    def _add_data(self, data):
        s = GroupedSubset(data, self)
        data.add_subset(s)
        self.subsets.append(s)

    def _remove_data(self, data):
        for s in list(self.subsets):
            if s.data is data:
                self.subsets.remove(s)

    def register_to_hub(self, hub):
        hub.subscribe(self, DataCollectionAddMessage,
                      lambda x: self._add_data(x.data))
        hub.subscribe(self, DataCollectionDeleteMessage,
                      lambda x: self._remove_data(x.data))
