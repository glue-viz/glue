from astropy import units as u

from glue.config import unit_converter
from glue.core import Subset


@unit_converter('simple-astropy')
class SimpleAstropyUnitConverter:

    def equivalent_units(self, data, cid):
        """
        Returns a list of units (as strings) equivalent to the ones for the
        data and component ID specified.
        """
        units = self._get_units(data, cid)
        if units:
            equivalent_units = u.Unit(units).find_equivalent_units(include_prefix_units=True)
            equivalent_units = map(equivalent_units, str)
        else:
            equivalent_units = []
        return equivalent_units

    def to_unit(self, data, cid, values, target_units):
        if target_units == '':
            return values
        original_units = self._get_units(data, cid)
        if original_units:
            return (values * u.Unit(original_units)).to_value(target_units)
        else:
            return values

    def to_native(self, data, cid, values, original_units):
        if original_units == '':
            return values
        target_units = self._get_units(data, cid)
        if target_units:
            return (values * u.Unit(original_units)).to_value(target_units)
        else:
            return values

    def _get_units(self, data, cid):
        data = data.data if isinstance(data, Subset) else data
        return data.get_component(cid).units


def get_default_unit_converter():
    # TODO: make the default customizable
    return unit_converter.members['simple-astropy']()


def find_unit_choices(unit_strings):
    equivalent_units = []
    for unit_string in sorted(unit_strings):
        try:
            if unit_string not in equivalent_units:
                equivalent_units.append(unit_string)
            for x in u.Unit(unit_string).find_equivalent_units(include_prefix_units=True):
                if x not in equivalent_units:
                    equivalent_units.append(str(x))
        except ValueError:
            pass
    return equivalent_units


def unit_scaling(original, target):
    return u.Unit(original).to(target)
