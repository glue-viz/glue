from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import numpy as np
from matplotlib.axes import Axes
from mock import MagicMock, patch
from numpy.testing import assert_array_equal

from glue.core.tests.test_state import clone
from glue.core.tests.util import simple_session
from glue.core.subset import SubsetState
from glue.core import Data
from glue import custom_viewer

from glue.app.qt import GlueApplication
from glue.app.qt.tests.test_application import check_clone_app

from ..custom_viewer import (FormElement, NumberElement,
                             ChoiceElement, CustomViewer,
                             CustomSubsetState, AttributeInfo,
                             FloatElement, TextBoxElement, SettingsOracle,
                             MissingSettingError, FrozenSettings)


def _make_widget(viewer):
    s = simple_session()
    return viewer._widget_cls(s)


viewer = custom_viewer('Testing Custom Viewer',
                       a=(0, 100),
                       b='att',
                       c='att(x)',
                       d=True,
                       e=False,
                       f=['a', 'b', 'c'],
                       g=OrderedDict(a=1, b=2, c=3),
                       h=64
                       )


setup = MagicMock()
settings_changed = MagicMock()
plot_subset = MagicMock()
plot_data = MagicMock()
make_selector = MagicMock()
make_selector.return_value = MagicMock(spec=SubsetState)
make_selector().copy.return_value = MagicMock(spec=SubsetState)
make_selector().copy().to_mask.return_value = np.array([False, True, True])


@viewer.setup
def _setup(axes):
    setup(axes)


@viewer.plot_data
def _plot_data(axes, a, b, g, h):
    plot_data(axes=axes, a=a, b=b, g=g, h=h)
    return []


@viewer.plot_subset
def _plot_subset(b, c, d, e, f, style):
    plot_subset(b=b, c=c, d=d, e=e, f=f, style=style)
    return []


@viewer.settings_changed
def _settings_changed(state):
    settings_changed(state=state)


@viewer.make_selector
def _make_selector(roi, c):
    make_selector(roi=roi, c=c)
    return SubsetState()


def test_custom_classes_dont_share_methods():
    """Regression test for #479"""
    a = custom_viewer('a')
    b = custom_viewer('b')
    assert a._custom_functions is not b._custom_functions


class ViewerSubclass(CustomViewer):
    a = (0, 100)
    b = 'att'
    c = 'att(x)'
    d = True
    e = False
    f = ['a', 'b', 'c']
    g = OrderedDict(a=1, b=2, c=3)
    h = 64

    def setup(self, axes):
        return setup(axes)

    def plot_data(self, axes, a, b, g, h):
        return plot_data(axes=axes, a=a, b=b, g=g, h=h)

    def plot_subset(self, b, c, d, e, f, style):
        return plot_subset(b=b, c=c, d=d, e=e, f=f, style=style)

    def settings_changed(self, state):
        return settings_changed(state=state)

    def make_selector(self, roi, c):
        return make_selector(roi=roi, c=c)


class TestCustomViewer(object):

    def setup_class(self):
        self.viewer = viewer

    def setup_method(self, method):
        setup.reset_mock()
        settings_changed.reset_mock()
        plot_subset.reset_mock()
        plot_data.reset_mock()
        make_selector.reset_mock()

        self.data = Data(x=[1, 2, 3], y=[2, 3, 4])
        self.session = simple_session()
        self.dc = self.session.data_collection
        self.dc.append(self.data)

    def teardown_method(self, method):
        if hasattr(self, 'w'):
            self.w.unregister(self.session.hub)

    def build(self):
        w = self.viewer._widget_cls(self.session)
        w.register_to_hub(self.session.hub)
        self.w = w
        return w

    def test_setup_called_on_init(self):
        ct = setup.call_count
        self.build()
        assert setup.call_count == ct + 1

    def test_separate_widgets_have_separate_state(self):
        w1 = self.build()
        w2 = self.build()

        assert w1._coordinator is not w2._coordinator
        assert w1._coordinator.state is not w2._coordinator.state

    def test_plot_data(self):
        w = self.build()
        w.add_data(self.data)

        a, k = plot_data.call_args
        assert isinstance(k['axes'], Axes)
        assert set(k.keys()) == set(('axes', 'a', 'b', 'g', 'h'))
        assert k['a'] == 50
        assert k['g'] == 1
        assert k['h'] == 64

    def test_plot_subset(self):
        w = self.build()
        w.add_data(self.data)

        self.dc.new_subset_group(subset_state=self.data.id['x'] > 2)

        a, k = plot_subset.call_args
        assert set(k.keys()) == set(('b', 'c', 'd', 'e', 'f', 'style'))

        assert_array_equal(k['b'].values, [3])
        assert_array_equal(k['c'].values, [3])
        assert k['d']
        assert not k['e']
        assert k['f'] == 'a'

    def test_make_selector(self):
        w = self.build()
        roi = MagicMock()
        w.apply_roi(roi)

        a, k = make_selector.call_args

        assert set(k.keys()) == set(('roi', 'c'))
        assert k['roi'] is roi

    def test_settings_change(self):
        w = self.build()
        ct = settings_changed.call_count
        w._coordinator._settings['d'].ui.setChecked(False)
        assert settings_changed.call_count == ct + 1
        a, k = settings_changed.call_args
        assert 'state' in k

    def test_register(self):
        with patch('glue.viewers.custom.qt.FormElement.register_to_hub') as r:
            self.build()
        assert r.call_count > 0

    def test_component(self):

        w = self.build()
        w.add_data(self.data)

        assert_array_equal(w._coordinator.value('b', layer=self.data).values,
                           [1, 2, 3])

    def test_component_autoupdate(self):

        w = self.build()
        w.add_data(self.data)

        assert w._coordinator._settings['b'].ui.count() == 2
        self.data.add_component([10, 20, 30], label='c')
        assert w._coordinator._settings['b'].ui.count() == 3

    def test_settings_changed_called_on_init(self):
        self.build()
        assert settings_changed.call_count == 1

    def test_selections_enabled(self):
        w = self.build()
        assert w._coordinator.selections_enabled
        assert 'select:rectangle' in w.toolbar.tools
        assert 'select:polygon' in w.toolbar.tools


def test_state_save():
    app = GlueApplication()
    w = app.new_data_viewer(viewer._widget_cls)  # noqa
    check_clone_app(app)


def test_state_save_with_data_layers():
    app = GlueApplication()
    dc = app.data_collection
    d = Data(x=[1, 2, 3], label='test')
    dc.append(d)
    w = app.new_data_viewer(viewer._widget_cls)
    w.add_data(d)
    check_clone_app(app)


class TestCustomSelectMethod(object):

    def setup_class(self):
        self.viewer = custom_viewer('CustomSelectViewer',
                                    x='att(x)', flip=False)

        @self.viewer.select
        def select(roi, x, flip):
            if flip:
                return x <= 1
            return x > 1

    def setup_method(self, method):
        self.data = Data(x=[1, 2, 3], y=[2, 3, 4])
        self.session = simple_session()
        self.dc = self.session.data_collection
        self.dc.append(self.data)

    def build(self):
        return self.viewer._widget_cls(self.session)

    def test_state(self):
        w = self.build()
        v = w._coordinator
        roi = MagicMock()
        s = CustomSubsetState(type(v), roi, v.settings())
        assert_array_equal(s.to_mask(self.data), [False, True, True])

    def test_state_view(self):
        w = self.build()
        v = w._coordinator
        roi = MagicMock()
        s = CustomSubsetState(type(v), roi, v.settings())

        assert_array_equal(s.to_mask(self.data, view=slice(None, None, 2)),
                           [False, True])

    def test_settings_frozen_at_creation(self):
        w = self.build()
        v = w._coordinator
        roi = MagicMock()
        s = CustomSubsetState(type(v), roi, v.settings())
        w.flip = True
        assert_array_equal(s.to_mask(self.data), [False, True, True])

    def test_save_load(self):
        w = self.build()
        v = w._coordinator
        roi = None
        s = CustomSubsetState(type(v), roi, v.settings())

        s2 = clone(s)

        assert_array_equal(s2.to_mask(self.data), [False, True, True])


class TestCustomViewerSubclassForm(TestCustomViewer):

    def setup_class(self):
        self.viewer = ViewerSubclass


class TestFormElements(object):

    def test_number_default_value(self):
        e = FormElement.auto((0, 100, 30))
        assert e.value() == 30

    def test_number_float(self):
        e = FormElement.auto((0.0, 1.0, 0.3))
        assert e.value() == 0.3

    def test_number_list(self):
        e = FormElement.auto([0, 10])
        assert isinstance(e, NumberElement)

    def test_choice_list(self):
        e = FormElement.auto(['a', 'b'])
        assert isinstance(e, ChoiceElement)

    def test_choice_tuple(self):
        e = FormElement.auto(('a', 'b'))
        assert isinstance(e, ChoiceElement)

    def test_float(self):
        e = FormElement.auto(1.2)
        assert isinstance(e, FloatElement)

        e = FormElement.auto(2)
        assert isinstance(e, FloatElement)
        assert e.value() == 2

    def test_textbox(self):
        e = FormElement.auto('_str')
        assert isinstance(e, TextBoxElement)
        assert e.value() == 'str'

    def test_recognizes_subsubclasses(self):

        class SubClassFormElement(TextBoxElement):
            @classmethod
            def recognizes(cls, params):
                return params == 'specific_class'

        e = FormElement.auto('specific_class')
        assert isinstance(e, SubClassFormElement)

    def test_unrecognized(self):
        with pytest.raises(ValueError):
            FormElement.auto(None)


class TestAttributeInfo(object):

    def setup_method(self, method):
        d = Data(x=[1, 2, 3, 4, 5], c=['a', 'b', 'a', 'a', 'b'], label='test')
        s = d.new_subset()
        s.subset_state = d.id['x'] > 2
        self.d = d
        self.s = s

    def test_numerical(self):
        v = AttributeInfo.from_layer(self.d, self.d.id['x'])
        assert_array_equal(v, [1, 2, 3, 4, 5])
        assert v.id == self.d.id['x']
        assert v.categories is None

    def test_categorical(self):
        v = AttributeInfo.from_layer(self.d, self.d.id['c'])
        assert_array_equal(v, [0, 1, 0, 0, 1])
        assert v.id == self.d.id['c']
        assert_array_equal(v.categories, ['a', 'b'])

    def test_subset(self):
        v = AttributeInfo.from_layer(self.s, self.d.id['x'])
        assert_array_equal(v, [3, 4, 5])
        assert v.id == self.d.id['x']
        assert v.categories is None

    def test_has_component(self):

        v = AttributeInfo.from_layer(self.s, self.d.id['x'])
        comp = self.s.data.get_component(self.d.id['x'])
        assert v._component == comp


class TestSettingsOracle(object):

    def test_oracle_raises_original_error(self):
        class BadFormElement(TextBoxElement):

            def value(self, layer=None, view=None):
                raise AttributeError('Inner Error')

        oracle = SettingsOracle({'bad_form': BadFormElement('str("text")')})

        try:
            oracle('bad_form')
            assert False
        except AttributeError as err:
            assert 'Inner Error' in err.args

    def test_oracle_raises_missing(self):
        oracle = SettingsOracle({'Form': TextBoxElement('_text')})
        with pytest.raises(MissingSettingError):
            oracle('missing')

    def test_frozen_oracle_raises_missing(self):

        oracle = FrozenSettings()
        with pytest.raises(MissingSettingError):
            oracle.value('missing')

    def test_load_reserved_words(self):

        _self = MagicMock()
        layer = MagicMock()
        style = layer.style
        extra = MagicMock()
        oracle = SettingsOracle({}, _self=_self,
                                layer=layer,
                                extra=extra)
        assert oracle('self') == _self
        assert oracle('layer') == layer
        assert oracle('style') == style
        assert oracle('extra') == extra

    def test_setting_names(self):

        oracle = SettingsOracle({'Form': TextBoxElement('_text')})
        assert sorted(oracle.setting_names()) == sorted(['style', 'layer', 'Form'])

    def test_raises_if_overlapping_reserved_words(self):

        with pytest.raises(AssertionError):
            SettingsOracle({'self': TextBoxElement('_text')})


def test_two_custom_viewer_classes():

    class MyWidget1(CustomViewer):

        text_box1_Widget1 = '_Hello'

        def setup(self, text_box1_Widget1):
            pass

    class MyWidget2(CustomViewer):

        text_box1_Widget2 = '_Hello'
        text_box2_Widget2 = '_world'

        def setup(self, text_box1_Widget2, text_box2_Widget2):
            pass

    app = GlueApplication()
    dc = app.data_collection
    d = Data(x=[1, 2, 3], label='test')
    dc.append(d)
    app.new_data_viewer(MyWidget1._widget_cls)
    app.new_data_viewer(MyWidget2._widget_cls)
