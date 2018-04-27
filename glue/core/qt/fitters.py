from __future__ import absolute_import, division, print_function

from qtpy import QtWidgets, QtGui

from glue.core.qt.simpleforms import build_form_item

__all__ = ['ConstraintsWidget', 'FitSettingsWidget']


class ConstraintsWidget(QtWidgets.QWidget):

    """
    A widget to display and tweak the constraints of a :class:`~glue.core.fitters.BaseFitter1D`
    """

    def __init__(self, constraints, parent=None):
        """
        Parameters
        ----------
        constraints : dict
            The `contstraints` property of a :class:`~glue.core.fitters.BaseFitter1D`
            object
        parent : QtWidgets.QWidget (optional)
            The parent of this widget
        """
        super(ConstraintsWidget, self).__init__(parent)
        self.constraints = constraints

        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(4)

        self.setLayout(self.layout)

        self.layout.addWidget(QtWidgets.QLabel("Estimate"), 0, 1)
        self.layout.addWidget(QtWidgets.QLabel("Fixed"), 0, 2)
        self.layout.addWidget(QtWidgets.QLabel("Bounded"), 0, 3)
        self.layout.addWidget(QtWidgets.QLabel("Lower Bound"), 0, 4)
        self.layout.addWidget(QtWidgets.QLabel("Upper Bound"), 0, 5)

        self._widgets = {}
        names = sorted(list(self.constraints.keys()))

        for k in names:
            row = []
            w = QtWidgets.QLabel(k)
            row.append(w)

            v = QtGui.QDoubleValidator()
            e = QtWidgets.QLineEdit()
            e.setValidator(v)
            e.setText(str(constraints[k]['value'] or ''))
            row.append(e)

            w = QtWidgets.QCheckBox()
            w.setChecked(constraints[k]['fixed'])
            fix = w
            row.append(w)

            w = QtWidgets.QCheckBox()
            limits = constraints[k]['limits']
            w.setChecked(limits is not None)
            bound = w
            row.append(w)

            e = QtWidgets.QLineEdit()
            e.setValidator(v)
            if limits is not None:
                e.setText(str(limits[0]))
            row.append(e)

            e = QtWidgets.QLineEdit()
            e.setValidator(v)
            if limits is not None:
                e.setText(str(limits[1]))
            row.append(e)

            def unset(w):
                def result(active):
                    if active:
                        w.setChecked(False)
                return result

            fix.toggled.connect(unset(bound))
            bound.toggled.connect(unset(fix))

            self._widgets[k] = row

        for i, row in enumerate(names, 1):
            for j, widget in enumerate(self._widgets[row]):
                self.layout.addWidget(widget, i, j)

    def settings(self, name):
        """ Return the constraints for a single model parameter """
        row = self._widgets[name]
        name, value, fixed, limited, lo, hi = row
        value = float(value.text()) if value.text() else None
        fixed = fixed.isChecked()
        limited = limited.isChecked()
        lo = lo.text()
        hi = hi.text()
        limited = limited and not ((not lo) or (not hi))
        limits = None if not limited else [float(lo), float(hi)]
        return dict(value=value, fixed=fixed, limits=limits)

    def update_constraints(self, fitter):
        """ Update the constraints in a :class:`~glue.core.fitters.BaseFitter1D`
            based on the settings in this widget
        """
        for name in self._widgets:
            s = self.settings(name)
            fitter.set_constraint(name, **s)


class FitSettingsWidget(QtWidgets.QDialog):

    def __init__(self, fitter, parent=None):
        super(FitSettingsWidget, self).__init__(parent)
        self.fitter = fitter

        self._build_form()
        self._connect()
        self.setModal(True)

    def _build_form(self):
        fitter = self.fitter

        l = QtWidgets.QFormLayout()
        options = fitter.options
        self.widgets = {}
        self.forms = {}

        for k in sorted(options):
            item = build_form_item(fitter, k)
            l.addRow(item.label, item.widget)
            self.widgets[k] = item.widget
            self.forms[k] = item  # need to prevent garbage collection

        constraints = fitter.constraints
        if constraints:
            self.constraints = ConstraintsWidget(constraints)
            l.addRow(self.constraints)
        else:
            self.constraints = None

        self.okcancel = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                                   QtWidgets.QDialogButtonBox.Cancel)
        l.addRow(self.okcancel)
        self.setLayout(l)

    def _connect(self):
        self.okcancel.accepted.connect(self.accept)
        self.okcancel.rejected.connect(self.reject)
        self.accepted.connect(self.update_fitter_from_settings)

    def update_fitter_from_settings(self):
        for k, v in self.widgets.items():
            setattr(self.fitter, k, v.value())
        if self.constraints is not None:
            self.constraints.update_constraints(self.fitter)
