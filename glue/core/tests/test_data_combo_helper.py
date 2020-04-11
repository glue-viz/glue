import pytest
from unittest.mock import MagicMock

from glue.core import Data, DataCollection
from glue.core.component_id import ComponentID
from echo.selection import SelectionCallbackProperty
from glue.core.state_objects import State
from glue.core.coordinates import IdentityCoordinates

from ..data_combo_helper import (ComponentIDComboHelper, ManualDataComboHelper,
                                 DataCollectionComboHelper)


def selection_choices(state, property):
    items = getattr(type(state), property).get_choice_labels(state)

    return ":".join(items).replace('Coordinate components', 'coord').replace('Main components', 'main').replace('Derived components', 'derived')


class ExampleState(State):
    combo = SelectionCallbackProperty()


def test_component_id_combo_helper():

    state = ExampleState()

    dc = DataCollection([])

    helper = ComponentIDComboHelper(state, 'combo', dc)

    assert selection_choices(state, 'combo') == ""

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    dc.append(data1)
    helper.append_data(data1)

    assert selection_choices(state, 'combo') == "x:y"

    data2 = Data(a=[1, 2, 3], b=['a', 'b', 'c'], label='data2',
                 coords=IdentityCoordinates(n_dim=1))

    dc.append(data2)
    helper.append_data(data2)

    assert selection_choices(state, 'combo') == "data1:x:y:data2:a:b"

    helper.categorical = False

    assert selection_choices(state, 'combo') == "data1:x:y:data2:a"

    helper.numeric = False

    assert selection_choices(state, 'combo') == "data1:data2"

    helper.categorical = True
    helper.numeric = True

    helper.pixel_coord = True
    assert selection_choices(state, 'combo') == "data1:main:x:y:coord:Pixel Axis 0 [x]:data2:main:a:b:coord:Pixel Axis 0 [x]"

    helper.world_coord = True
    assert selection_choices(state, 'combo') == "data1:main:x:y:coord:Pixel Axis 0 [x]:data2:main:a:b:coord:Pixel Axis 0 [x]:World 0"

    helper.pixel_coord = False
    assert selection_choices(state, 'combo') == "data1:main:x:y:data2:main:a:b:coord:World 0"

    helper.world_coord = False

    dc.remove(data2)

    assert selection_choices(state, 'combo') == "x:y"

    data1['z'] = data1.id['x'] + 1

    assert selection_choices(state, 'combo') == "main:x:y:derived:z"

    helper.derived = False

    assert selection_choices(state, 'combo') == "x:y"

    data1.id['x'].label = 'z'
    assert selection_choices(state, 'combo') == "z:y"

    helper.remove_data(data1)

    assert selection_choices(state, 'combo') == ""


def test_component_id_combo_helper_nocollection():

    # Make sure that we can use use ComponentIDComboHelper without any
    # data collection.

    state = ExampleState()

    data = Data(x=[1, 2, 3], y=[2, 3, 4], z=['a', 'b', 'c'], label='data1')

    helper = ComponentIDComboHelper(state, 'combo', data=data)

    assert selection_choices(state, 'combo') == "x:y:z"

    helper.categorical = False

    assert selection_choices(state, 'combo') == "x:y"

    helper.numeric = False

    assert selection_choices(state, 'combo') == ""

    helper.categorical = True

    assert selection_choices(state, 'combo') == "z"

    helper.numeric = True

    assert selection_choices(state, 'combo') == "x:y:z"

    data2 = Data(a=[1, 2, 3], b=['a', 'b', 'c'], label='data2')

    with pytest.raises(Exception) as exc:
        helper.append_data(data2)
    assert exc.value.args[0] == ("Cannot change data in ComponentIDComboHelper "
                                 "initialized from a single dataset")

    with pytest.raises(Exception) as exc:
        helper.remove_data(data2)
    assert exc.value.args[0] == ("Cannot change data in ComponentIDComboHelper "
                                 "initialized from a single dataset")

    with pytest.raises(Exception) as exc:
        helper.set_multiple_data([data2])
    assert exc.value.args[0] == ("Cannot change data in ComponentIDComboHelper "
                                 "initialized from a single dataset")


def test_component_id_combo_helper_init():

    # Regression test to make sure that the numeric and categorical options
    # in the __init__ are taken into account properly

    state = ExampleState()

    dc = DataCollection([])

    data = Data(a=[1, 2, 3], b=['a', 'b', 'c'], label='data2')
    dc.append(data)

    helper = ComponentIDComboHelper(state, 'combo', dc)
    helper.append_data(data)
    assert selection_choices(state, 'combo') == "a:b"

    helper = ComponentIDComboHelper(state, 'combo', dc, numeric=False)
    helper.append_data(data)
    assert selection_choices(state, 'combo') == "b"

    helper = ComponentIDComboHelper(state, 'combo', dc, categorical=False)
    helper.append_data(data)
    assert selection_choices(state, 'combo') == "a"

    helper = ComponentIDComboHelper(state, 'combo', dc, numeric=False, categorical=False)
    helper.append_data(data)
    assert selection_choices(state, 'combo') == ""


def test_component_id_combo_helper_replaced():

    # Make sure that when components are replaced, the equivalent combo index
    # remains selected and an event is broadcast so that any attached callback
    # properties can be sure to pull the latest text/userData.

    callback = MagicMock()

    state = ExampleState()
    state.add_callback('combo', callback)

    dc = DataCollection([])

    helper = ComponentIDComboHelper(state, 'combo', dc)

    assert selection_choices(state, 'combo') == ""

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    callback.reset_mock()

    dc.append(data1)
    helper.append_data(data1)

    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert selection_choices(state, 'combo') == "x:y"

    new_id = ComponentID(label='new')

    data1.update_id(data1.id['x'], new_id)

    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert selection_choices(state, 'combo') == "new:y"


def test_component_id_combo_helper_add():

    # Make sure that when adding a component, and if a data collection is not
    # present, the choices still get updated

    callback = MagicMock()

    state = ExampleState()
    state.add_callback('combo', callback)

    dc = DataCollection([])

    helper = ComponentIDComboHelper(state, 'combo')

    assert selection_choices(state, 'combo') == ""

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    callback.reset_mock()

    dc.append(data1)
    helper.append_data(data1)

    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert selection_choices(state, 'combo') == "x:y"

    data1.add_component([7, 8, 9], 'z')

    # Should get notification since choices have changed
    callback.assert_called_once_with(0)
    callback.reset_mock()

    assert selection_choices(state, 'combo') == "x:y:z"


@pytest.mark.parametrize('initialize_data_collection', [False, True])
def test_manual_data_combo_helper(initialize_data_collection):

    # The case with initialize_data_collection=False is a regression test for a
    # bug which meant that when a ManualDataComboHelper was initialized without
    # a data collection, it did not change when a data object added later has a
    # label changed.

    callback = MagicMock()
    state = ExampleState()
    state.add_callback('combo', callback)

    dc = DataCollection([])

    if initialize_data_collection:
        helper = ManualDataComboHelper(state, 'combo', dc)
    else:
        helper = ManualDataComboHelper(state, 'combo')

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    dc.append(data1)

    assert callback.call_count == 0

    assert selection_choices(state, 'combo') == ""

    helper.append_data(data1)
    assert callback.call_count == 1

    assert selection_choices(state, 'combo') == "data1"

    data1.label = 'mydata1'
    assert selection_choices(state, 'combo') == "mydata1"
    assert callback.call_count == 2

    if initialize_data_collection:

        dc.remove(data1)

        assert selection_choices(state, 'combo') == ""
        assert callback.call_count == 3


def test_data_collection_combo_helper():

    callback = MagicMock()
    state = ExampleState()
    state.add_callback('combo', callback)

    dc = DataCollection([])

    helper = DataCollectionComboHelper(state, 'combo', dc)  # noqa

    data1 = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')

    assert callback.call_count == 0

    dc.append(data1)

    assert callback.call_count == 1

    assert selection_choices(state, 'combo') == "data1"

    data1.label = 'mydata1'
    assert selection_choices(state, 'combo') == "mydata1"

    assert callback.call_count == 2

    dc.remove(data1)

    assert callback.call_count == 3

    assert selection_choices(state, 'combo') == ""


def test_component_id_combo_helper_rename():

    # Make sure that renaming component IDs now propagates to the combo options

    state = ExampleState()

    data = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
    dc = DataCollection([data])

    helper = ComponentIDComboHelper(state, 'combo', dc)  # noqa
    helper.append_data(data)

    assert selection_choices(state, 'combo') == "x:y"

    data.id['x'].label = 'renamed'

    assert selection_choices(state, 'combo') == "renamed:y"


def test_component_id_combo_helper_reorder():

    # Make sure that renaming component IDs now propagates to the combo options

    state = ExampleState()

    data = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
    dc = DataCollection([data])

    helper = ComponentIDComboHelper(state, 'combo', dc)  # noqa
    helper.append_data(data)

    assert selection_choices(state, 'combo') == "x:y"

    data.reorder_components(data.components[::-1])

    assert selection_choices(state, 'combo') == "y:x"


def test_component_id_combo_helper_none():

    # Make sure the none=True option works

    state = ExampleState()

    data = Data(x=[1, 2, 3], y=[2, 3, 4], label='data1')
    dc = DataCollection([data])

    helper = ComponentIDComboHelper(state, 'combo', dc)
    helper.append_data(data)

    assert selection_choices(state, 'combo') == "x:y"

    helper.none = True

    assert selection_choices(state, 'combo') == ":x:y"

    helper.none = 'banana'

    assert selection_choices(state, 'combo') == "banana:x:y"

    # Try with initializing none=... from the start

    helper = ComponentIDComboHelper(state, 'combo', dc, none=True)
    helper.append_data(data)
    assert selection_choices(state, 'combo') == ":x:y"

    helper = ComponentIDComboHelper(state, 'combo', dc, none='banana')
    helper.append_data(data)
    assert selection_choices(state, 'combo') == "banana:x:y"
