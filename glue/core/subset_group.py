"""
A :class:`~glue.core.subset_group.SubsetGroup` unites a group of
:class:`~glue.core.subset.Subset` instances together with a consistent state,
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
from __future__ import absolute_import, division, print_function

from warnings import warn

from glue.external import six
from glue.core.contracts import contract
from glue.core.message import (DataCollectionAddMessage,
                               DataCollectionDeleteMessage)
from glue.core.visual import VisualAttributes
from glue.core.hub import HubListener
from glue.utils import Pointer
from glue.core.subset import SubsetState
from glue.core import Subset
from glue.config import settings


__all__ = ['GroupedSubset', 'SubsetGroup']


class GroupedSubset(Subset):

    """
    A member of a SubsetGroup, whose internal representation
    is shared with other group members
    """
    subset_state = Pointer('group.subset_state')
    label = Pointer('group.label')

    def __init__(self, data, group):
        """
        :param data: :class:`~glue.core.data.Data` instance to bind to
        :param group: :class:`~glue.core.subset_group.SubsetGroup`
        """
        self.group = group
        super(GroupedSubset, self).__init__(data, label=group.label,
                                            color=group.style.color,
                                            alpha=group.style.alpha)

    def _setup(self, color, alpha, label):
        self.color = color
        self.label = label  # trigger disambiguation
        self.style = VisualAttributes(parent=self)
        self.style.markersize *= 2.5
        self.style.color = color
        self.style.alpha = alpha
        # skip state setting here

    @property
    def verbose_label(self):
        return "%s (%s)" % (self.label, self.data.label)

    def sync_style(self, other):
        self.style.set(other)

    def __eq__(self, other):
        return other is self

    # In Python 3, if __eq__ is defined, then __hash__ has to be re-defined
    if six.PY3:
        __hash__ = object.__hash__

    def __gluestate__(self, context):
        return dict(group=context.id(self.group),
                    style=context.do(self.style))

    @classmethod
    def __setgluestate__(cls, rec, context):
        dummy_grp = SubsetGroup()  # __init__ needs group.label
        self = cls(None, dummy_grp)
        yield self
        self.group = context.object(rec['group'])
        self.style = context.object(rec['style'])


class SubsetGroup(HubListener):

    def __init__(self, color=settings.SUBSET_COLORS[0], alpha=0.5, label=None, subset_state=None):
        """
        Create a new empty SubsetGroup

        Note: By convention, SubsetGroups should be created via
        DataCollection.new_subset.
        """
        self.subsets = []
        if subset_state is None:
            subset_state = SubsetState()

        self.subset_state = subset_state
        self.label = label
        self._style = None

        self.style = VisualAttributes(parent=self)
        self.style.markersize *= 2.5
        self.style.color = color
        self.style.alpha = alpha

    @contract(data='isinstance(DataCollection)')
    def register(self, data):
        """
        Register to a :class:`~glue.core.data_collection.DataCollection`

        This is called automatically by
        :meth:`glue.core.data_collection.DataCollection.new_subset_group`
        """
        self.register_to_hub(data.hub)

        # add to self, then register, so fully populated by first
        # broadcast

        for d in data:
            s = GroupedSubset(d, self)
            self.subsets.append(s)

        for d, s in zip(data, self.subsets):
            d.add_subset(s)

    def paste(self, other_subset):
        """paste subset state from other_subset onto self """
        state = other_subset.subset_state.copy()
        self.subset_state = state

    def _add_data(self, data):
        # add a new data object to group
        s = GroupedSubset(data, self)
        data.add_subset(s)
        self.subsets.append(s)

    def _remove_data(self, data):
        # remove a data object from group
        for s in list(self.subsets):
            if s.data is data:
                self.subsets.remove(s)

    def register_to_hub(self, hub):
        hub.subscribe(self, DataCollectionAddMessage,
                      lambda x: self._add_data(x.data))
        hub.subscribe(self, DataCollectionDeleteMessage,
                      lambda x: self._remove_data(x.data))

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
        self._style = value
        self._sync_style()

    def _sync_style(self):
        for s in self.subsets:
            s.sync_style(self.style)

    @contract(item='string')
    def broadcast(self, item):
        # used by __setattr__ and VisualAttributes.__setattr__
        if item == 'style':
            self._sync_style()
            return

        for s in self.subsets:
            s.broadcast(item)

    def __setattr__(self, attr, value):
        object.__setattr__(self, attr, value)
        if attr in ['subset_state', 'label', 'style']:
            self.broadcast(attr)

    def __gluestate__(self, context):
        return dict(label=self.label,
                    state=context.id(self.subset_state),
                    style=context.do(self.style),
                    subsets=list(map(context.id, self.subsets)))

    @classmethod
    def __setgluestate__(cls, rec, context):
        result = cls()
        yield result
        result.subset_state = context.object(rec['state'])
        result.label = rec['label']
        result.style = context.object(rec['style'])
        result.style.parent = result
        result.subsets = list(map(context.object, rec['subsets']))

    def __and__(self, other):
        return self.subset_state & other.subset_state

    def __or__(self, other):
        return self.subset_state | other.subset_state

    def __xor__(self, other):
        return self.subset_state ^ other.subset_state

    def __invert__(self):
        return ~self.subset_state


def coerce_subset_groups(collect):
    """
    If necessary, reassign non-grouped subsets in a DataCollection
    into SubsetGroups.

    This is used to support DataCollections saved with
    version 1 of glue.core.state.save_data_collection
    """
    for data in collect:
        for subset in data.subsets:
            if not isinstance(subset, GroupedSubset):
                warn("DataCollection has subsets outside of "
                     "subset groups, which are no longer supported. "
                     "Moving to subset groups")
                subset.delete()
                grp = collect.new_subset_group()
                grp.subset_state = subset.subset_state
                grp.style = subset.style
                grp.label = subset.label
