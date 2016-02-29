import numpy as np

from glue.external.qt import QtGui
from glue.core import Subset
from glue.utils.qt import update_combobox
from glue.utils.qt.widget_properties import (CurrentComboTextProperty,
                                             CurrentComboProperty,
                                             FloatLineProperty)


__all__ = ['AttributeLimitsHelper']


class AttributeLimitsHelper(object):
    """
    This class is a helper for attribute-dependent min/max level values.

    Given an attribute combo as well as line edit widgets for the min/max
    values, this helper takes care of populating the attribute combo, setting
    the initial values of the min/max values, and keeping a cache of the
    min/max values as a function of attribute. This means that if the user
    edits the min/max values and then changes attribute then changes back, the
    original min/max values will be retained.

    In addition, this helper class can optionally link a combo for the scale
    mode, for example using the min/max values or percentile values, as well as
    a button for flipping the min/max values.

    Parameters
    ----------
    attribute_combo : ``QComboBox`` instance
        The attribute combo - this will be populated once a dataset is assigned
        to the helper.
    lower_value, upper_value : ``QLineEdit`` instances
        The fields for the lower/upper levels
    mode_combo : ``QComboBox`` instance, optional
        The scale mode combo - this will be populated by presets such as
        Min/Max, various percentile levels, and Custom.
    flip_button : ``QToolButton`` instance, optional
        The flip button
    data : :class:`glue.core.data.Data`
        The dataset to attach to the helper - this will be used to populate the
        attribute combo as well as determine the limits automatically given the
        scale mode preset.

    Notes
    -----

    Once the helper is instantiated, the data associated with the helper can be
    set/changed with:

    >>> helper = AttributeLimitsHelper(...)
    >>> helper.data = data

    The data can also be passed to the initializer as described in the list of
    parameters above.
    """

    attribute = CurrentComboProperty('attribute_combo')
    scale_mode = CurrentComboTextProperty('mode_combo')
    percentile = CurrentComboProperty('mode_combo')
    vlo = FloatLineProperty('lower_value')
    vhi = FloatLineProperty('upper_value')

    def __init__(self, attribute_combo, lower_value, upper_value,
                       mode_combo=None, flip_button=None, data=None):

        self.attribute_combo = attribute_combo
        self.mode_combo = mode_combo
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.flip_button = flip_button

        self.attribute_combo.currentIndexChanged.connect(self._update_limits)

        self.lower_value.editingFinished.connect(self._manual_edit)
        self.upper_value.editingFinished.connect(self._manual_edit)

        if self.mode_combo is None:
            # Make hidden combo box to avoid having to always figure out if the
            # combo mode exists. This will then always be set to Min/Max.
            self.mode_combo = QtGui.QComboBox()

        self._setup_mode_combo()
        self.mode_combo.currentIndexChanged.connect(self._update_mode)

        if self.flip_button is not None:
            self.flip_button.clicked.connect(self._flip_limits)

        self._limits = {}
        self._callbacks = []

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._setup_attribute_combo()

    def _setup_attribute_combo(self):
        self.attribute_combo.clear()
        if isinstance(self.data, Subset):
            components = self.data.data.visible_components
        else:
            components = self.data.visible_components
        label_data = [(comp.label, comp) for comp in components]
        update_combobox(self.attribute_combo, label_data)

    def _setup_mode_combo(self):
        self.mode_combo.clear()
        self.mode_combo.addItem("Min/Max", userData=100)
        self.mode_combo.addItem("99.5%", userData=99.5)
        self.mode_combo.addItem("99%", userData=99)
        self.mode_combo.addItem("95%", userData=95)
        self.mode_combo.addItem("90%", userData=90)
        self.mode_combo.addItem("Custom", userData=None)
        self.mode_combo.setCurrentIndex(-1)

    def _flip_limits(self):
        self.vlo, self.vhi = self.vhi, self.vlo

    def _manual_edit(self):
        self._cache_limits()

    def _update_mode(self):
        if self.scale_mode == 'Custom':
            self.lower_value.setEnabled(True)
            self.upper_value.setEnabled(True)
        else:
            self.lower_value.setEnabled(False)
            self.upper_value.setEnabled(False)
            self._auto_limits()
            self._cache_limits()

    def _cache_limits(self):
        self._limits[self.attribute] = self.scale_mode, self.vlo, self.vhi

    def _update_limits(self):
        if self.attribute in self._limits:
            self.lower_value.blockSignals(True)
            self.scale_mode, self.vlo, self.vhi = self._limits[self.attribute]
            self.lower_value.blockSignals(False)
        else:
            self.scale_mode = 'Min/Max'
            self._update_mode()

    def _auto_limits(self):

        exclude = (100 - self.percentile) / 2.

        # For subsets, we want to compute the limits based on the full dataset,
        # not just the subset.
        if isinstance(self.data, Subset):
            data_values = self.data.data[self.attribute]
        else:
            data_values = self.data[self.attribute]

        self.vlo = np.nanpercentile(data_values, exclude)
        self.vhi = np.nanpercentile(data_values, 100 - exclude)
