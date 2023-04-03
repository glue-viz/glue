from glue.core.units import UnitConverter, find_unit_choices
from glue.config import unit_converter, settings
from glue.core import Data


def setup_function(func):
    func.ORIGINAL_UNIT_CONVERTER = settings.UNIT_CONVERTER


def teardown_function(func):
    settings.UNIT_CONVERTER = func.ORIGINAL_UNIT_CONVERTER


def setup_module():
    unit_converter.add('test-custom', SimpleCustomUnitConverter)


def teardown_module():
    unit_converter._members.pop('test-custom')


def test_unit_converter_default():

    data1 = Data(a=[1, 2, 3], b=[4, 5, 6])
    data1.get_component('a').units = 'm'

    uc = UnitConverter()
    assert 'km' in uc.equivalent_units(data1, data1.id['a'])

    assert uc.to_unit(data1, data1.id['a'], 2000, 'km') == 2
    assert uc.to_native(data1, data1.id['a'], 2, 'km') == 2000

    assert uc.to_unit(data1, data1.id['a'], 2000, None) == 2000
    assert uc.to_native(data1, data1.id['a'], 2, None) == 2

    assert uc.equivalent_units(data1, data1.id['b']) == []

    assert uc.to_unit(data1, data1.id['b'], 2000, 'km') == 2000
    assert uc.to_native(data1, data1.id['b'], 2, 'km') == 2


def test_find_unit_choices_default():

    assert find_unit_choices([]) == []

    units1 = find_unit_choices([(None, None, 'm')])
    assert 'km' in units1
    assert 'yr' not in units1

    units2 = find_unit_choices([(None, None, 'm'), (None, None, 's')])
    assert 'km' in units2
    assert 'yr' in units2


class SimpleCustomUnitConverter:

    def equivalent_units(self, data, cid, units):
        # We want to make sure we properly test data and cid so we make it
        # so that if cid contains 'fixed' we return only the original unit
        # and if the data label contains 'bilingual' then we return the full
        # set of units
        if cid.label == 'fixed':
            return [units]
        elif data.label == 'bilingual':
            return ['one', 'two', 'three', 'dix', 'vingt', 'trente']
        elif units in ['one', 'two', 'three']:
            return ['one', 'two', 'three']
        elif units in ['dix', 'vingt', 'trente']:
            return ['dix', 'vingt', 'trente']
        else:
            raise ValueError(f'Unrecognized unit: {units}')

    numerical = {
        'one': 1,
        'two': 2,
        'three': 3,
        'dix': 10,
        'vingt': 20,
        'trente': 30
    }

    def to_unit(self, data, cid, values, original_units, target_units):
        return values * self.numerical[target_units] / self.numerical[original_units]


def test_unit_converter_custom():

    settings.UNIT_CONVERTER = 'test-custom'

    data1 = Data(a=[1, 2, 3])
    data1.get_component('a').units = 'two'

    uc = UnitConverter()
    assert uc.equivalent_units(data1, data1.id['a']) == ['one', 'two', 'three']

    assert uc.to_unit(data1, data1.id['a'], 4, 'three') == 6
    assert uc.to_native(data1, data1.id['a'], 6, 'three') == 4


def test_find_unit_choices_custom():

    settings.UNIT_CONVERTER = 'test-custom'

    data1 = Data(fixed=[1, 2, 3], a=[2, 3, 4], b=[3, 4, 5], label='data1')
    data2 = Data(c=[4, 5, 6], d=[5, 6, 7], label='bilingual')

    assert find_unit_choices([]) == []

    assert find_unit_choices([(data1, data1.id['fixed'], 'one')]) == ['one']
    assert find_unit_choices([(data1, data1.id['a'], 'one')]) == ['one', 'two', 'three']
    assert find_unit_choices([(data1, data1.id['a'], 'dix')]) == ['dix', 'vingt', 'trente']
    assert find_unit_choices([(data2, data2.id['c'], 'one')]) == ['one', 'two', 'three', 'dix', 'vingt', 'trente']
