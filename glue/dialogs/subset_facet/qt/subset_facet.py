import os
import numpy as np
from matplotlib import cm

from qtpy import QtWidgets
from glue.core.util import colorize_subsets, facet_subsets, subset_categorical
from glue.core.state_objects import State
from echo import CallbackProperty, SelectionCallbackProperty
from glue.utils.qt import load_ui
from glue.core.data_combo_helper import DataCollectionComboHelper, ComponentIDComboHelper
from echo.qt import autoconnect_callbacks_to_qt
from glue.core.state_objects import StateAttributeLimitsHelper

__all__ = ['SubsetFacetDialog', 'SubsetCategoricalsDialog']


class SubsetFacetState(State):

    log = CallbackProperty(False)
    v_min = CallbackProperty(0.)
    v_max = CallbackProperty(1.)
    steps = CallbackProperty(5)
    data = SelectionCallbackProperty()
    att = SelectionCallbackProperty()
    cmap = CallbackProperty()

    def __init__(self, data_collection):

        super(SubsetFacetState, self).__init__()

        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)
        self.att_helper = ComponentIDComboHelper(self, 'att')
        self.lim_helper = StateAttributeLimitsHelper(self, attribute='att',
                                                     lower='v_min', upper='v_max',
                                                     log='log')

        self.add_callback('data', self._on_data_change)
        self._on_data_change()

    def _on_data_change(self, *args, **kwargs):
        self.att_helper.set_multiple_data([] if self.data is None else [self.data])


class SubsetFacetDialog(QtWidgets.QDialog):
    """
    Create a new dialog for subset faceting

    Parameters
    ----------
    collect : :class:`~glue.core.data_collection.DataCollection`
        The data collection to use
    default : :class:`~glue.core.data.Data`, optional
        The default dataset in the collection (optional)
    """

    def __init__(self, collect, default=None, parent=None):

        super(SubsetFacetDialog, self).__init__(parent=parent)

        self.state = SubsetFacetState(collect)

        self.ui = load_ui('subset_facet.ui', self,
                          directory=os.path.dirname(__file__))
        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)

        self._collect = collect

        if default is not None:
            self.state.data = default

        self.state.cmap = cm.RdYlBu

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

    def _apply(self):

        try:
            lo, hi = self.state.v_min, self.state.v_max
        except ValueError:
            return  # limits not set. Abort

        if not np.isfinite(lo) or not np.isfinite(hi):
            return

        subsets = facet_subsets(self._collect, self.state.att, lo=lo, hi=hi,
                                steps=self.state.steps, log=self.state.log)
        colorize_subsets(subsets, self.state.cmap)

    @classmethod
    def facet(cls, collect, default=None, parent=None):
        """
        Class method to create faceted subsets.

        The arguments are the same as __init__.
        """
        self = cls(collect, parent=parent, default=default)
        value = self.exec_()

        if value == QtWidgets.QDialog.Accepted:
            self._apply()

class SubsetCategoricalsState(State):
            
    data = SelectionCallbackProperty()
    att = SelectionCallbackProperty()
    cmap = CallbackProperty()

    def __init__(self, data_collection):

        super(SubsetCategoricalsState, self).__init__()

        self.data_helper = DataCollectionComboHelper(self, 'data', data_collection)
        self.att_helper = ComponentIDComboHelper(self, 'att', numeric=True, categorical=True) #Some categoricals may be encoded as integers

        self.add_callback('data', self._on_data_change)
        self._on_data_change()

    def _on_data_change(self, *args, **kwargs):
        self.att_helper.set_multiple_data([] if self.data is None else [self.data])


class SubsetCategoricalsDialog(QtWidgets.QDialog):
    """
    Create a new dialog to create subsets on a categorical component

    Parameters
    ----------
    collect : :class:`~glue.core.data_collection.DataCollection`
        The data collection to use
    default : :class:`~glue.core.data.Data`, optional
        The default dataset in the collection (optional)
    """

    def __init__(self, collect, default=None, parent=None):

        super(SubsetCategoricalsDialog, self).__init__(parent=parent)

        self.state = SubsetCategoricalsState(collect)

        self.ui = load_ui('subset_categorical.ui', self,
                          directory=os.path.dirname(__file__))
        self._connections = autoconnect_callbacks_to_qt(self.state, self.ui)

        self._collect = collect

        if default is not None:
            self.state.data = default

        self.state.cmap = cm.tab20

        self.ui.button_ok.clicked.connect(self.accept)
        self.ui.button_cancel.clicked.connect(self.reject)

    def _apply(self):

        subsets = subset_categorical(self._collect, self.state.data, self.state.att)
        #colorize_subsets(subsets, self.state.cmap)

    @classmethod
    def facet(cls, collect, default=None, parent=None):
        """
        Class method to create faceted subsets.

        The arguments are the same as __init__.
        """
        self = cls(collect, parent=parent, default=default)
        value = self.exec_()

        if value == QtWidgets.QDialog.Accepted:
            self._apply()
