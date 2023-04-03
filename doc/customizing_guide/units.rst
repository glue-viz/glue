Unit conversion in glue
=======================

.. note:: Support for automatic unit conversion in glue is experimental - at the moment
          the ability to select units for the x and y axes is only available in the profile viewer.

Data components can be assigned units as a string (or `None` to indicate no known units).
By default, glue uses `astropy.units <https://docs.astropy.org/en/stable/units/index.html>`_ package
to carry out unit conversions based on these units. However, it is possible to customize the
unit conversion machinery, either to use a different unit transformation machinery, or to specify,
e.g., equivalencies in the astropy unit conversion. To customize the unit conversion behavior, you
will need to define a unit converter as shown below::

    from astropy import units as u
    from glue.core.units import unit_converter

    @unit_converter('custom-name')
    class MyCustomUnitConverter:

        def equivalent_units(self, data, cid, units):
            # Given a glue data object (data), a component ID (cid), and the units
            # of that component in the data object (units), this method should
            # return a flat list of units (as strings) that the data could be
            # converted to. This is used to construct the drop-down menus with the
            # available units to convert to.

        def to_unit(self, data, cid, values, original_units, target_units):
            # Given a glue data object (data), a component ID (cid), the values
            # to convert, and the original and target units of the values, this method
            # should return the converted values. Note that original_units
            # gives the units of the values array, which might not be the same
            # as the original native units of the component in the data.

In both methods, the data and cid are passed in not to get values or units (those should be
used from the other arguments to the methods) but rather to allow logic for the unit
conversion that might depend on which component is being converted. An example of
a simple unit converter based on `astropy.units`_ would be::

    from astropy import units as u
    from glue.core.units import unit_converter

    @unit_converter('example-1')
    class ExampleUnitConverter:

        def equivalent_units(self, data, cid, units):
            return map(str, u.Unit(units).find_equivalent_units(include_prefix_units=True))

        def to_unit(self, data, cid, values, original_units, target_units):
            return (values * u.Unit(original_units)).to_value(target_units)

This does not actually make use of ``data`` and ``cid``. An example that does would be::

    from astropy import units as u
    from glue.core.units import unit_converter

    @unit_converter('example-2')
    class ExampleUnitConverter:

        def equivalent_units(self, data, cid, units):
            equivalencies = u.temperature() if 'temp' in cid.label.lower() else None
            return map(str, u.Unit(units).find_equivalent_units(include_prefix_units=True, equivalencies=equivalencies))

        def to_unit(self, data, cid, values, original_units, target_units):
            equivalencies = u.temperature() if 'temp' in cid.label.lower() else None
            return (values * u.Unit(original_units)).to_value(target_units, equivalencies=equivalencies)

Once you have defined a unit conversion class, you can then opt-in to using it in glue by adjusting
the following setting::

    from glue.config import settings
    settings.UNIT_CONVERTER = 'example-2'
