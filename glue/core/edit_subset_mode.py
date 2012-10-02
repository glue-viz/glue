"""These classes define the behavior of how new subset states affect
    the edit_subset of a Data object.

   The EditSubsetMode is universal in Glue -- all datasets and clients
   share the same mode. This is enforced by making the base
   EditSubsetMode object a singleton.
"""
#pylint: disable=I0011, R0903
from .decorators import singleton
from .data import Data
from .data_collection import DataCollection

@singleton
class EditSubsetMode(object):
    """ Implements how new SubsetStates modify the edit_subset state """
    def __init__(self):
        self.mode = ReplaceMode

    def _combine_data(self, data, new_state):
        """ Dispatches to the combine method of mode attribute.

        The behavior is dependent on the mode it dispatches to.
        By default, the method uses ReplaceMode, which overwrites
        the edit_subset's subset_state with new_state

        :param edit_subset: The current edit_subset
        :param new_state: The new SubsetState
        """
        if len(data.subsets) == 0:
            data.edit_subset = data.new_subset()
        if data.edit_subset is None:
            return
        subs = data.edit_subset
        if not isinstance(subs, list):
            subs = [subs]
        for s in subs:
            self.mode(s, new_state)

    def combine(self, d, new_state):
        """ Apply a new subset state to editable subsets within a
        :class:`~glue.core.data.Data` or
        :class:`~glue.core.data_collection.DataCollection` instance

        :param d: Data or Collection to act upon
        :type d: Data or DataCollection

        :param new_state: Subset state to combine with
        :type new_state: :class:`~glue.core.subset.SubsetState`
        """
        if isinstance(d, Data):
            self._combine_data(d, new_state)
        elif isinstance(d, DataCollection):
            for data in d:
                self._combine_data(data, new_state)
        else:
            raise TypeError("input must be a Data or DataCollection: %s" %
                            type(d))


def ReplaceMode(edit_subset, new_state):
    """ Replaces edit_subset.subset_state with new_state """
    edit_subset.subset_state = new_state


def AndMode(edit_subset, new_state):
    """ Edit_subset.subset state is and-combined with new_state """
    new_state.parent = edit_subset
    state = new_state & edit_subset.subset_state
    edit_subset.subset_state = state


def OrMode(edit_subset, new_state):
    """ Edit_subset.subset state is or-combined with new_state """
    new_state.parent = edit_subset
    state = new_state | edit_subset.subset_state
    edit_subset.subset_state = state


def XorMode(edit_subset, new_state):
    """ Edit_subset.subset state is xor-combined with new_state """
    new_state.parent = edit_subset
    state = new_state ^ edit_subset.subset_state
    edit_subset.subset_state = state


def AndNotMode(edit_subset, new_state):
    """ Edit_subset.subset state is and-not-combined with new_state """
    new_state.parent = edit_subset
    state = edit_subset.subset_state & (~new_state)
    edit_subset.subset_state = state
