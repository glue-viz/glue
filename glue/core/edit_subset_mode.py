"""These classes define the behavior of how new subset states affect
    the edit_subset of a Data object.

   The EditSubsetMode is universal in Glue -- all datasets and clients
   share the same mode. This is enforced by making the base
   EditSubsetMode object a singleton.
"""
# pylint: disable=I0011, R0903

from __future__ import absolute_import, division, print_function

import logging

from glue.core.contracts import contract
from glue.core.data_collection import DataCollection
from glue.core.data import Data
from glue.core.decorators import singleton
from glue.utils import as_list


@singleton
class EditSubsetMode(object):

    """ Implements how new SubsetStates modify the edit_subset state """

    def __init__(self):
        self.mode = ReplaceMode
        self.data_collection = None

    def _combine_data(self, data, new_state, add_if_empty=False):
        """ Dispatches to the combine method of mode attribute.

        The behavior is dependent on the mode it dispatches to.
        By default, the method uses ReplaceMode, which overwrites
        the edit_subsets' subset_state with new_state

        :param edit_subset: The current edit_subset
        :param new_state: The new SubsetState
        :param add_if_empty: If True and a data set has no subsets,
                             a new one will be added and assigned
                             using new-state
        """
        empty = data.edit_subset is None or data.edit_subset == []
        if add_if_empty and empty:
            if self.data_collection is None:
                raise RuntimeError("Must set data_collection before "
                                   "calling update")
            data.edit_subset = self.data_collection.new_subset_group()
        if empty and not add_if_empty:
            logging.getLogger(__name__).info("Ignoring subset update")
            return
        subs = data.edit_subset
        for s in as_list(subs):
            self.mode(s, new_state)

    @contract(d='inst($DataCollection, $Data)',
              new_state='isinstance(SubsetState)',
              focus_data='inst($Data)|None')
    def update(self, d, new_state, focus_data=None):
        """ Apply a new subset state to editable subsets within a
        :class:`~glue.core.data.Data` or
        :class:`~glue.core.data_collection.DataCollection` instance

        :param d: Data or Collection to act upon
        :type d: Data or DataCollection

        :param new_state: Subset state to combine with
        :type new_state: :class:`~glue.core.subset.SubsetState`

        :param focus_data: The main data set in focus by the client,
        if relevant. If a data set is in focus and has no subsets,
        a new one will be created using new_state.
        """
        logging.getLogger(__name__).debug("Update subset for %s", d)

        if isinstance(d, Data):
            self._combine_data(d, new_state, add_if_empty=d is focus_data)
        elif isinstance(d, DataCollection):
            no_editable = all(data.edit_subset is None or
                              data.edit_subset == []
                              for data in d)
            for data in d:
                doadd = data is focus_data and no_editable
                self._combine_data(data, new_state, add_if_empty=doadd)
        else:
            raise TypeError("input must be a Data or DataCollection: %s" %
                            type(d))


def ReplaceMode(edit_subset, new_state):
    """ Replaces edit_subset.subset_state with new_state """
    logging.getLogger(__name__).debug("Replace %s", edit_subset)
    edit_subset.subset_state = new_state.copy()


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
