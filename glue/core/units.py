from astropy import units as u


def find_unit_choices(unit_strings):
    equivalent_units = []
    for unit_string in sorted(unit_strings):
        try:
            equivalent_units.append(unit_string)
            equivalent_units.extend([str(x) for x in u.Unit(unit_string).find_equivalent_units()])
        except ValueError:
            pass
    return equivalent_units


def unit_scaling(original, target):
    return u.Unit(original).to(target)
