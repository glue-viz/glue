from qtpy import QtWidgets, QtGui

from glue.external.echo.qt.autoconnect import autoconnect_callbacks_to_qt
from glue.external.echo import CallbackProperty
from glue.external.echo.qt.connect import UserDataWrapper


def test_autoconnect_callbacks_to_qt():

    class Data(object):
        pass

    data1 = Data()
    data2 = Data()

    class CustomWidget(QtWidgets.QWidget):
        def __init__(self, parent=None):

            super(CustomWidget, self).__init__(parent=parent)

            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)

            self.combotext_planet = QtWidgets.QComboBox(objectName='combotext_planet')
            self.layout.addWidget(self.combotext_planet)
            self.combotext_planet.addItem('earth')
            self.combotext_planet.addItem('mars')
            self.combotext_planet.addItem('jupiter')

            self.combodata_dataset = QtWidgets.QComboBox(objectName='combodata_dataset')
            self.layout.addWidget(self.combodata_dataset)
            self.combodata_dataset.addItem('data1', UserDataWrapper(data1))
            self.combodata_dataset.addItem('data2', UserDataWrapper(data2))

            self.text_name = QtWidgets.QLineEdit(objectName='text_name')
            self.layout.addWidget(self.text_name)

            self.valuetext_age = QtWidgets.QLineEdit(objectName='valuetext_age')
            self.layout.addWidget(self.valuetext_age)

            self.value_height = QtWidgets.QSlider(objectName='value_height')
            self.value_height.setMinimum(0)
            self.value_height.setMaximum(10)
            self.layout.addWidget(self.value_height)

            self.bool_log = QtWidgets.QToolButton(objectName='bool_log')
            self.bool_log.setCheckable(True)
            self.layout.addWidget(self.bool_log)

    class Person(object):
        planet = CallbackProperty()
        dataset = CallbackProperty()
        name = CallbackProperty()
        age = CallbackProperty()
        height = CallbackProperty()
        log = CallbackProperty()

    widget = CustomWidget()

    person = Person()

    connect_kwargs = {'height': {'value_range': (0, 100)},
                      'age': {'fmt':'{:.2f}'}}

    connections = autoconnect_callbacks_to_qt(person, widget, connect_kwargs=connect_kwargs)

    # Check that modifying things in the Qt widget updates the callback properties

    widget.combotext_planet.setCurrentIndex(2)
    assert person.planet == 'jupiter'

    widget.combodata_dataset.setCurrentIndex(1)
    assert person.dataset is data2

    widget.text_name.setText('Lovelace')
    widget.text_name.editingFinished.emit()
    assert person.name == 'Lovelace'

    widget.valuetext_age.setText('76')
    widget.valuetext_age.editingFinished.emit()
    assert person.age == 76

    widget.value_height.setValue(7)
    assert person.height == 70

    widget.bool_log.setChecked(True)
    assert person.log

    # Check that modifying the callback properties updates the Qt widget

    person.planet = 'mars'
    assert widget.combotext_planet.currentIndex() == 1

    person.dataset = data1
    assert widget.combodata_dataset.currentIndex() == 0

    person.name = 'Curie'
    assert widget.text_name.text() == 'Curie'

    person.age = 66.3
    assert widget.valuetext_age.text() == '66.30'

    person.height = 54
    assert widget.value_height.value() == 5

    person.log = False
    assert not widget.bool_log.isChecked()

def test_autoconnect_with_empty_qt_item():

    # The following test just makes sure that if a widget without children
    # is ever passed to autoconnect_callbacks_to_qt, things don't crash

    widget = QtGui.QPalette()

    class Person(object):
        name = CallbackProperty()

    person = Person()

    autoconnect_callbacks_to_qt(person, widget)
