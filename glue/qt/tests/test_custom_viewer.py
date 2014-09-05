import pytest
from mock import MagicMock, patch
from numpy.testing import assert_array_equal
from matplotlib.axes import Axes

from ... import custom_viewer
from ...core import Data
from ...core.tests.util import simple_session
from ..custom_viewer import FormElement, NumberElement, ChoiceElement


def _make_widget(viewer):
    s = simple_session()
    return viewer._widget_cls(s)


class TestCustomViewer(object):

    def setup_class(self):

        self.viewer = custom_viewer('Custom Viewer',
                                    a=(0, 100),
                                    b='att',
                                    c='att(x)',
                                    d=True,
                                    e=False,
                                    f=['a', 'b', 'c'],
                                    g=dict(a=1, b=2, c=3),
                                    )

        self.setup = self.viewer.setup(MagicMock())
        self.update_settings = self.viewer.update_settings(MagicMock())
        self.plot_subset = self.viewer.plot_subset(MagicMock())
        self.plot_data = self.viewer.plot_data(MagicMock())
        self.make_selector = self.viewer.make_selector(MagicMock())

    def setup_method(self, method):
        self.setup.reset_mock()
        self.update_settings.reset_mock()
        self.plot_subset.reset_mock()
        self.plot_data.reset_mock()
        self.make_selector.reset_mock()

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
        self.build()
        assert self.setup.call_count == 1

    def test_settings(self):
        w = self.build()
        s = w.settings(self.data)

        assert s['a'] == 50
        assert_array_equal(s['c'].values, [1, 2, 3])
        assert s['d'] is True
        assert s['e'] is False
        assert s['f'] == 'a'
        assert s['g'] == 1

    def test_plot_data(self):
        w = self.build()
        w.add_data(self.data)

        a, k = self.plot_data.call_args
        assert isinstance(a[0], Axes)
        assert set(k.keys()) == set(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'style'))
        assert_array_equal(k['c'].values, [1, 2, 3])

    def test_plot_subset(self):
        w = self.build()
        w.add_data(self.data)

        self.dc.new_subset_group(subset_state=self.data.id['x'] > 2)

        a, k = self.plot_subset.call_args
        assert set(k.keys()) == set(('a', 'b', 'c', 'd', 'e', 'f', 'g', 'style'))
        assert_array_equal(k['c'].values, [3])

    def test_make_selector(self):
        w = self.build()
        roi = MagicMock()
        self.make_selector.return_value = self.data.id['x'] > 1
        w.client.apply_roi(roi)

        a, k = self.make_selector.call_args

        assert a == (roi,)
        assert set(k.keys()) == set(('a', 'b', 'c', 'd', 'e', 'f', 'g'))
        assert k['d'] is True

    def test_settings_change(self):
        w = self.build()
        w._settings['d'].ui.setChecked(False)

        assert self.update_settings.call_count == 1

        assert w.settings()['d'] is False

    def test_register(self):
        with patch('glue.qt.custom_viewer.FormElement.register_to_hub') as r:
            w = self.build()
        assert r.call_count > 0

    def test_component(self):

        w = self.build()
        w.add_data(self.data)

        assert_array_equal(w.settings(self.data)['b'].values, [1, 2, 3])

    def test_component_autoupdate(self):

        w = self.build()
        w.add_data(self.data)

        assert w._settings['b'].ui.count() == 2
        self.data.add_component([10, 20, 30], label='c')
        assert w._settings['b'].ui.count() == 3


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

    def test_unrecognized(self):
        with pytest.raises(ValueError):
            e = FormElement.auto(None)
