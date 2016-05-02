# pylint: disable=I0011,W0613,W0201,W0212,E1101,E1103

from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
from mock import MagicMock

from .. import parse
from ..data import ComponentID, Component, Data
from ..subset import Subset


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
        assert expected == result

    def test_reference_list_invalid_cmd(self):
        with pytest.raises(KeyError) as exc:
            parse._reference_list('{a}', {})
        assert exc.value.args[0] == ("Tags from command not in "
                                     "reference mapping")

    def test_dereference(self):
        c1 = ComponentID('c1')
        c2 = ComponentID('c2')
        s1 = Subset(None, label='s1')
        s2 = Subset(None, label='s2')
        refs = dict([('c1', c1), ('c2', c2), ('s1', s1), ('s2', s2)])
        cmd = '({c1} > 10) and {s1}'
        expected = ('(data[references["c1"], __view] > 10) and '
                    'references["s1"].to_mask(__view)')
        result = parse._dereference(cmd, refs)
        assert expected == result

    def test_validate(self):
        ref = {'a': 1, 'b': 2}
        parse._validate('{a} + {b}', ref)
        parse._validate('{a}', ref)
        parse._validate('3 + 4', ref)
        with pytest.raises(parse.InvalidTagError) as exc:
            parse._validate('{c}', ref)
        assert exc.value.args[0] == ("Tag c not in reference mapping: "
                                     "['a', 'b']")

    def test_ensure_only_component_references(self):
        ref = {'a': 1, 'b': ComponentID('b')}
        F = parse._ensure_only_component_references
        F('{b} + 5', ref)
        with pytest.raises(TypeError) as exc:
            F('{b} + {a}', ref)
        assert exc.value.args[0] == ("Reference to a, which is not a "
                                     "ComponentID")
        with pytest.raises(TypeError) as exc:
            F('{b} + {d}', ref)
        assert exc.value.args[0] == ("Reference to d, which is not a "
                                     "ComponentID")


class TestParsedCommand(object):

    def test_evaluate_component(self):
        data = MagicMock()
        c1 = ComponentID('c1')
        data.__getitem__.return_value = 5
        cmd = '{comp1} * 5'
        refs = {'comp1': c1}
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 25
        data.__getitem__.assert_called_once_with((c1, None))

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
        data.__getitem__.assert_called_once_with((c1, None))

    def test_evaluate_math(self):

        # If numpy, np, and math aren't defined in the config.py file, they
        # are added to the local variables available.

        data = MagicMock()
        c1 = ComponentID('c1')
        data.__getitem__.return_value = 10
        refs = {'comp1': c1}

        cmd = 'numpy.log10({comp1})'
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 1

        cmd = 'np.log10({comp1})'
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 1

        cmd = 'math.log10({comp1})'
        pc = parse.ParsedCommand(cmd, refs)
        assert pc.evaluate(data) == 1

    def test_evaluate_test(self):

        data = MagicMock()
        c1 = ComponentID('c1')
        data.__getitem__.return_value = 10
        refs = {'comp1': c1}

        cmd = 'numpy.log10({comp1}) + 3.4 - {comp1}'
        pc = parse.ParsedCommand(cmd, refs)
        pc.evaluate_test()

        cmd = 'nump.log10({comp1}) + 3.4 - {comp1}'
        pc = parse.ParsedCommand(cmd, refs)
        with pytest.raises(NameError) as exc:
            pc.evaluate_test()
        assert exc.value.args[0] == "name 'nump' is not defined"


class TestParsedComponentLink(object):

    def make_link(self):
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
        return data, c2

    def test(self):
        data, cid = self.make_link()
        result = data[cid]
        expected = np.array([100, 200, 300])
        np.testing.assert_array_equal(result, expected)

    def test_not_identity(self):
        # regression test
        d = Data(x=[1, 2, 3])
        c2 = ComponentID('c2')
        cmd = '{x}'
        refs = {'x': d.id['x']}
        pc = parse.ParsedCommand(cmd, refs)
        link = parse.ParsedComponentLink(c2, pc)
        assert not link.identity

    def test_slice(self):
        data, cid = self.make_link()
        result = data[cid, ::2]
        np.testing.assert_array_equal(result, [100, 300])

    def test_save_load(self):
        from .test_state import clone

        d = Data(x=[1, 2, 3])
        c2 = ComponentID('c2')
        cmd = '{x} + 1'
        refs = {'x': d.id['x']}
        pc = parse.ParsedCommand(cmd, refs)
        link = parse.ParsedComponentLink(c2, pc)
        d.add_component_link(link)

        d2 = clone(d)
        np.testing.assert_array_equal(d2['c2'], [2, 3, 4])


class TestParsedSubsetState(object):

    def setup_method(self, method):
        data = Data(g=[2, 4, 6, 8])

        s1 = data.new_subset()
        s2 = data.new_subset()

        s1.subset_state = np.array([1, 1, 1, 0], dtype=bool)
        s2.subset_state = np.array([0, 1, 1, 1], dtype=bool)

        self.refs = {'s1': s1, 's2': s2, 'g': data.id['g']}
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
