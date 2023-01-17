from astropy import units as u

from glue.config import unit_converter, settings
from glue.core import Subset

__all__ = ['UnitConverter', 'find_unit_choices']


class UnitConverter:

    def __init__(self):
        self.converter_helper = unit_converter.members[settings.UNIT_CONVERTER]()

    def equivalent_units(self, data, cid):
        """
        Returns a list of units (as strings) equivalent to the ones for the
        data and component ID specified.
        """
        units = self._get_units(data, cid)
        if units:
            equivalent_units = self.converter_helper.equivalent_units(data, cid, units)
        else:
            equivalent_units = []
        return equivalent_units

    def to_unit(self, data, cid, values, target_units):
        if target_units is None:
            return values
        original_units = self._get_units(data, cid)
        if original_units:
            return self.converter_helper.to_unit(data, cid, values, original_units, target_units)
        else:
            return values

    def to_native(self, data, cid, values, original_units):
        if original_units is None:
            return values
        target_units = self._get_units(data, cid)
        if target_units:
            return self.converter_helper.to_unit(data, cid, values, original_units, target_units)
        else:
            return values

    def _get_units(self, data, cid):
        data = data.data if isinstance(data, Subset) else data
        return data.get_component(cid).units


@unit_converter('default')
class SimpleAstropyUnitConverter:

    def equivalent_units(self, data, cid, units):
        return map(str, u.Unit(units).find_equivalent_units(include_prefix_units=True))

    def to_unit(self, data, cid, values, original_units, target_units):
        return (values * u.Unit(original_units)).to_value(target_units)


def find_unit_choices(data_cid_units):
    equivalent_units = []
    converter_helper = unit_converter.members[settings.UNIT_CONVERTER]()
    for data, cid, unit_string in data_cid_units:
        try:
            if unit_string not in equivalent_units:
                equivalent_units.append(unit_string)
            for x in converter_helper.equivalent_units(data, cid, unit_string):
                if x not in equivalent_units:
                    equivalent_units.append(str(x))
        except ValueError:
            pass
    return equivalent_units
