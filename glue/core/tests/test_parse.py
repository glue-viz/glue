import pytest
import numpy as np
from mock import MagicMock

from ..data import ComponentID, Component, Data
from ..subset import Subset
from .. import parse


class TestParse(object):

    def test_re_matches_valid_names(self):
        reg = parse.TAG_RE
        valid = ['{a}', '{ a }', '{A}', '{a }', '{ a}',
                 '{a_}', '{abc_1}', '{_abc_1}', '{1}', '{1_}']
        invalid = ['', '{}', '{a b}']
        for v in valid:
            assert reg.match(v) is not None
        for i in invalid:
            assert reg.match(i) is None

    def test_group(self):
        reg = parse.TAG_RE
        assert reg.match('{a}').group('tag') == 'a'
        assert reg.match('{ a }').group('tag') == 'a'
        assert reg.match('{ A }').group('tag') == 'A'
        assert reg.match('{ Abc_ }').group('tag') == 'Abc_'

    def test_reference_list(self):
        cmd = '{a} - {b} + {c}'
        refs = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        expected = set([1, 2, 3])
        result = set(parse._reference_list(cmd, refs))
        assert  expected == result

    def test_reference_list_invalid_cmd(self):
        with pytest.raises(KeyError):
            parse._reference_list('{a}', {})

    def test_dereference(self):
        c1 = ComponentID('c1')
        c2 = ComponentID('c2')
        s1 = Subset(None, label='s1')
        s2 = Subset(None, label='s2')
        refs = dict([('c1', c1), ('c2', c2), ('s1', s1), ('s2', s2)])
        cmd = '({c1} > 10) and {s1}'
        expected = ('(data[references["c1"]] > 10) and '
                    'references["s1"].to_mask()')
        result = parse._dereference(cmd, refs)
        assert expected == result

    def test_validate(self):
        ref = {'a': 1, 'b': 2}
        parse._validate('{a} + {b}', ref)
        parse._validate('{a}', ref)
        parse._validate('3 + 4', ref)
        with pytest.raises(TypeError):
            parse._validate('{c}', ref)

    def test_ensure_only_component_references(self):
        ref = {'a': 1, 'b': ComponentID('b')}
        F = parse._ensure_only_component_references
        F('{b} + 5', ref)
        with pytest.raises(TypeError):
            F('{b} + {a}', ref)
        with pytest.raises(TypeError):
            F('{b} + {d}', ref)


class TestParsedCommand(object):

    def test_evaluate_component(self):
        data = MagicMock()
        c1 = ComponentID('c1')
        data.__getitem__.return_value = 5
        cmd = '{comp1} * 5'
        refs = {'comp1': c1}
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 25
        data.__getitem__.assert_called_once_with(c1)

    def test_evaluate_subset(self):
        sub = MagicMock(spec_set=Subset)
        sub2 = MagicMock(spec_set=Subset)
        sub.to_mask.return_value = 3
        sub2.to_mask.return_value = 4
        cmd = '{s1} and {s2}'
        refs = {'s1': sub, 's2': sub2}
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(None) == (3 and 4)

    def test_evaluate_function(self):
        data = MagicMock()
        c1 = ComponentID('c1')
        data.__getitem__.return_value = 5
        cmd = 'max({comp1}, 100)'
        refs = {'comp1': c1}
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 100
        data.__getitem__.assert_called_once_with(c1)


class TestParsedComponentLink(object):

    def test(self):
        data = Data()
        comp = Component(np.array([1, 2, 3]))
        c1 = ComponentID('c1')
        c2 = ComponentID('c2')
        data.add_component(comp, c1)

        cmd = '{comp1} * 100'
        refs = {'comp1': c1}
        pc = parse.ParsedCommand(cmd, refs)

        cl = parse.ParsedComponentLink(c2, pc)
        data.add_component_link(cl)

        result = data[c2]
        expected = np.array([100, 200, 300])
        np.testing.assert_array_equal(result, expected)


class TestParsedSubsetState(object):
    def setup_method(self, method):
        data = Data()
        c1 = Component(np.array([2, 4, 6, 8]))
        g = ComponentID('g')
        data.add_component(c1, g)

        s1 = data.new_subset()
        s2 = data.new_subset()

        state1 = MagicMock()
        state1.to_mask.return_value = np.array([1, 1, 1, 0], dtype=bool)
        state2 = MagicMock()
        state2.to_mask.return_value = np.array([0, 1, 1, 1], dtype=bool)

        s1.subset_state = state1
        s2.subset_state = state2

        self.refs = {'s1': s1, 's2': s2, 'g': g}
        self.data = data

    def test_two_subset(self):
        cmd = '{s1} & {s2}'
        s = self.data.new_subset()
        p = parse.ParsedCommand(cmd, self.refs)
        state = parse.ParsedSubsetState(p)

        s.subset_state = state

        result = s.to_mask()
        expected = np.array([0, 1, 1, 0], dtype=bool)

        np.testing.assert_array_equal(result, expected)

    def test_two_subset_and_component(self):
        cmd = '{s1} & {s2} & ({g} < 6)'
        s = self.data.new_subset()
        p = parse.ParsedCommand(cmd, self.refs)
        state = parse.ParsedSubsetState(p)

        s.subset_state = state
        result = s.to_mask()
        expected = np.array([0, 1, 0, 0], dtype=bool)

        np.testing.assert_array_equal(result, expected)
