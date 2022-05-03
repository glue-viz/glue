from astropy import units as u


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
