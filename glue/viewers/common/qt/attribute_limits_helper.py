import numpy as np

from glue.external.qt import QtGui
from glue.core import Subset
from glue.utils.qt import update_combobox
from glue.utils.qt.widget_properties import (CurrentComboTextProperty,
                                             CurrentComboProperty,
                                             FloatLineProperty)

# TODO: there is room for optimization here, in particular to ensure that
# signals are emitted the absolute minimum of times.


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
        self._invalidate_cache()
        self._data = value
        if isinstance(value, Subset):
            self.subset_mode = 'data'
        else:
            self.subset_mode = None
        self._setup_attribute_combo()

    def set_limits(self, vlo, vhi):
        self.lower_value.blockSignals(True)
        self.upper_value.blockSignals(True)
        self.vlo = vlo
        self.vhi = vhi
        self.lower_value.blockSignals(False)
        self.upper_value.blockSignals(False)
        self.lower_value.editingFinished.emit()
        self.upper_value.editingFinished.emit()

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
        self.set_limits(self.vhi, self.vlo)

    def _manual_edit(self):
        self._cache_limits()

    def _update_mode(self):
        if self.scale_mode != 'Custom':
            self._auto_limits()
            self._cache_limits()

    def _invalidate_cache(self):
        self._limits.clear()

    def _cache_limits(self):
        if self.subset_mode != 'outline':
            self._limits[self.attribute] = self.scale_mode, self.vlo, self.vhi

    def _update_limits(self):
        if self.subset_mode == 'outline':
            self.set_limits(0, 2)
        elif self.attribute in self._limits:
            self.scale_mode, lower, upper = self._limits[self.attribute]
            self.set_limits(lower, upper)
        else:
            self.mode_combo.blockSignals(True)
            self.scale_mode = 'Min/Max'
            self.mode_combo.blockSignals(False)
            self._auto_limits()

    def _auto_limits(self):

        if self.data is None:
            return

        if self.attribute is None:
            return

        if self.subset_mode == 'outline':
            self.set_limits(0, 2)
            return

        exclude = (100 - self.percentile) / 2.

        # For subsets in 'data' mode, we want to compute the limits based on
        # the full dataset, not just the subset.
        if self.subset_mode == 'data':
            data_values = self.data.data[self.attribute]
        else:
            data_values = self.data[self.attribute]

        lower = np.nanpercentile(data_values, exclude)
        upper = np.nanpercentile(data_values, 100 - exclude)

        if self.subset_mode == 'data':
            self.set_limits(0, upper)
        else:
            self.set_limits(lower, upper)

    @property
    def subset_mode(self):
        return self._subset_mode

    @subset_mode.setter
    def subset_mode(self, value):

        if isinstance(self.data, Subset):
            if value not in ['outline', 'data']:
                raise ValueError("subset_mode should either be 'outline', 'data' when data is a subset")

            self.lower_value.setEnabled(False)

            if value == 'outline':
                self.attribute_combo.setEnabled(False)
                self.mode_combo.setEnabled(False)
                self.upper_value.setEnabled(False)
            else:
                self.attribute_combo.setEnabled(True)
                self.mode_combo.setEnabled(True)
                self.upper_value.setEnabled(True)

            self.flip_button.setEnabled(False)

        else:

            if value is not None:
                raise ValueError("subset_mode should be set to None when data is not a subset")

            self.attribute_combo.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.lower_value.setEnabled(True)
            self.upper_value.setEnabled(True)
            self.flip_button.setEnabled(True)

        self._subset_mode = value

        self._update_limits()
