"""These classes define the behavior of how new subset states affect
    the edit_subset of a Data object.

   The EditSubsetMode is universal in Glue -- all datasets and clients share
   the same mode. This is enforced by making the base EditSubsetMode object a singleton.
"""
from .decorators import singleton


@singleton
class EditSubsetMode(object):
    """ Implements how new SubsetStates modify the edit_subset state """
    def __init__(self):
        self.mode = ReplaceMode()

    def combine(self, edit_subset, new_state):
        """ Dispatches to the combine method of mode attribute.

        The behavior is dependent on the mode it dispatches to.
        By default, the method uses ReplaceMode, which overwrites
        the edit_subset's subset_state with new_state

        :param edit_subset: The current edit_subset
        :param new_state: The new SubsetState
        """
        self.mode.combine(edit_subset, new_state)


class ReplaceMode(object):
    def combine(self, edit_subset, new_state):
        """ Replaces edit_subset.subset_state with new_state """
        edit_subset.subset_state = new_state


class AndMode(object):
    def combine(self, edit_subset, new_state):
        """ Edit_subset.subset state is and-combined with new_state """
        state = new_state & edit_subset.subset_state
        edit_subset.subset_state = state


class OrMode(object):
    def combine(self, edit_subset, new_state):
        """ Edit_subset.subset state is or-combined with new_state """
        state = new_state | edit_subset.subset_state
        edit_subset.subset_state = state


class XorMode(object):
    def combine(self, edit_subset, new_state):
        """ Edit_subset.subset state is xor-combined with new_state """
        state = new_state ^ edit_subset.subset_state
        edit_subset.subset_state = state


class AndNotMode(object):
    def combine(self, edit_subset, new_state):
        """ Edit_subset.subset state is and-not-combined with new_state """
        state = edit_subset.subset_state & (~new_state)
        edit_subset.subset_state = state


class SpawnMode(object):
    def combine(self, edit_subset, new_state):
        """new_state is set to edit_subset.subset_state, and current
        state is added as new subset
        """
        sub = edit_subset.data.new_subset()
        sub.subset_state = edit_subset.subset_state
        edit_subset.subset_state = new_state
