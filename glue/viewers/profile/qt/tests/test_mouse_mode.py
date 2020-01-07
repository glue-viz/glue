from unittest.mock import MagicMock

from matplotlib import pyplot as plt

from ..mouse_mode import NavigateMouseMode, RangeMouseMode


def test_navigate_mouse_mode():

    callback = MagicMock()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 10)
    viewer = MagicMock()
    viewer.axes = ax
    mode = NavigateMouseMode(viewer, press_callback=callback)

    event = MagicMock()
    event.xdata = 1.5
    event.inaxes = True
    mode.press(event)
    assert mode.state.x is None
    mode.move(event)
    assert mode.state.x is None
    mode.release(event)
    assert mode.state.x is None
    mode.activate()
    mode.press(event)
    assert callback.call_count == 1
    assert mode.state.x == 1.5
    event.xdata = 2.5
    mode.move(event)
    assert mode.state.x == 2.5
    mode.release(event)
    event.xdata = 3.5
    mode.move(event)
    assert mode.state.x == 2.5
    mode.deactivate()
    event.xdata = 1.5
    mode.press(event)
    assert callback.call_count == 1
    assert mode.state.x == 2.5
    mode.activate()
    event.xdata = 3.5
    mode.press(event)
    assert callback.call_count == 2
    assert mode.state.x == 3.5
    event.inaxes = False
    event.xdata = 4.5
    mode.press(event)
    assert callback.call_count == 2
    assert mode.state.x == 3.5


def test_range_mouse_mode():

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 10)
    viewer = MagicMock()
    viewer.axes = ax
    mode = RangeMouseMode(viewer)

    event = MagicMock()
    event.xdata = 1.5
    event.inaxes = True

    # Pressing, moving, and releasing doesn't do anything until mode is active
    mode.press(event)
    assert mode.state.x_min is None
    assert mode.state.x_max is None
    mode.move(event)
    assert mode.state.x_min is None
    assert mode.state.x_max is None
    mode.release(event)
    assert mode.state.x_min is None
    assert mode.state.x_max is None

    mode.activate()

    # Click and drag then creates an interval where x_min is the first value
    # that was clicked and x_max is set to the position of the mouse while
    # dragging and at the point of releasing.

    mode.press(event)
    assert mode.state.x_min == 1.5
    assert mode.state.x_max == 1.5

    event.xdata = 2.5
    mode.move(event)
    assert mode.state.x_min == 1.5
    assert mode.state.x_max == 2.5

    event.xdata = 3.5
    mode.move(event)
    assert mode.state.x_min == 1.5
    assert mode.state.x_max == 3.5

    mode.release(event)
    event.xdata = 4.5
    mode.move(event)
    assert mode.state.x_min == 1.5
    assert mode.state.x_max == 3.5

    # Test that we can drag the existing edges by clicking on then

    event.xdata = 1.49
    mode.press(event)
    event.xdata = 1.25
    mode.move(event)
    assert mode.state.x_min == 1.25
    assert mode.state.x_max == 3.5
    mode.release(event)

    event.xdata = 3.51
    mode.press(event)
    event.xdata = 4.0
    mode.move(event)
    assert mode.state.x_min == 1.25
    assert mode.state.x_max == 4.0
    mode.release(event)

    # Test that we can drag the entire interval by clicking inside

    event.xdata = 2
    mode.press(event)
    event.xdata = 3
    mode.move(event)
    assert mode.state.x_min == 2.25
    assert mode.state.x_max == 5.0
    mode.release(event)

    # Test that x_range works

    assert mode.state.x_range == (2.25, 5.0)

    # Clicking outside the range starts a new interval

    event.xdata = 6
    mode.press(event)
    event.xdata = 7
    mode.move(event)
    assert mode.state.x_min == 6
    assert mode.state.x_max == 7
    mode.release(event)

    # Deactivate and activate again to make sure that code for hiding/showing
    # artists gets executed

    mode.deactivate()

    event.xdata = 8
    mode.press(event)
    assert mode.state.x_min == 6
    assert mode.state.x_max == 7

    mode.activate()

    event.xdata = 9
    mode.press(event)
    event.xdata = 10
    mode.move(event)
    assert mode.state.x_min == 9
    assert mode.state.x_max == 10

    # Check that events outside the axes get ignored

    event.inaxes = False
    event.xdata = 11
    mode.press(event)
    assert mode.state.x_min == 9
    assert mode.state.x_max == 10
