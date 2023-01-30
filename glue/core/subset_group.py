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
:func:`glue.core.data_collection.DataCollection.new_subset_group`.
It should *not* call :func:`~glue.core.data.BaseData.add_subset` or
:func:`~glue.core.data.BaseData.new_subset` directly
"""

import uuid
from warnings import warn

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

    Parameters
    ----------
    data : :class:`~glue.core.data.Data`
        Instance to bind to.
    group : :class:`~glue.core.subset_group.SubsetGroup`
    """
    subset_state = Pointer('group.subset_state')
    label = Pointer('group.label')

    def __init__(self, data, group):
        # We deliberately don't call Subset.__init__ here because we don't want
        # to set e.g. the subset state, color, transparency, etc. Instead we
        # just want to defer to the SubsetGroup for these.

        self._broadcasting = False  # must be first def

        self.group = group

        self.data = data
        self.label = group.label  # trigger disambiguation

        # We assign a UUID which can then be used for example in equations
        # for derived components - the idea is that this doesn't change over
        # the life cycle of glue, so it is a more reliable way to refer to
        # components in strings than using labels
        self._uuid = str(uuid.uuid4())

    @property
    def style(self):
        return self.group.style

    @property
    def subset_group(self):
        return self.group.subset_group

    @property
    def verbose_label(self):
        return "%s (%s)" % (self.label, self.data.label)

    def __eq__(self, other):
        return other is self

    # If __eq__ is defined, then __hash__ has to be re-defined
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


class SubsetGroup(HubListener):

    def __init__(self, label=None, subset_state=None, **kwargs):
        """
        Create a new empty SubsetGroup

        Note: By convention, SubsetGroups should be created via
        :func:`glue.core.data_collection.DataCollection.new_subset_group`
        """
        self.subsets = []

        if subset_state is None:
            self.subset_state = SubsetState()
        else:
            self.subset_state = subset_state

        self.label = label

        visual_args = {k: v for k, v in kwargs.items() if k in VisualAttributes.DEFAULT_ATTS}
        visual_args.setdefault("color", settings.SUBSET_COLORS[0])
        visual_args.setdefault("alpha", 0.5)
        visual_args.setdefault("linewidth", 2.5)
        visual_args.setdefault("markersize", 7)

        self.style = VisualAttributes(parent=self, **visual_args)

    @contract(data='isinstance(DataCollection)')
    def register(self, data):
        """
        Register to a :class:`~glue.core.data_collection.DataCollection`

        This is called automatically by
        :func:`glue.core.data_collection.DataCollection.new_subset_group`
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
        """Paste subset state from other_subset onto self"""
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

    @contract(item='string')
    def broadcast(self, item):
        for s in self.subsets:
            s.broadcast(item)

    def __setattr__(self, attr, value):
        # We terminate early here if the value is not None but hasn't changed
        # to avoid broadcasting a message after.
        if value is not None and value == getattr(self, attr, None):
            return
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
    If necessary, reassign non-grouped subsets in a
    :class:`~glue.core.data_collection.DataCollection` into
    :class:`glue.core.subset_group.SubsetGroup`s.

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
