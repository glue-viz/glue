"""
An interface to PyContracts, to annotate functions with type information

The @contract decorator is disabled by default, to avoid
any runtime overhead when using Glue. To enable runtime
checking, run

glue.config.enable_contracts(True)

in a glue config file.

If PyContrats is imported, a no-op @contract decorator
is provided for compatibility

Glue code should only import contract through this module,
and never directly from the contracts package.
"""

from __future__ import absolute_import, division, print_function

from pandas import Series
from numpy import ndarray, s_

from glue.external.six import string_types
from glue.config import enable_contracts


def _build_custom_contracts():
    """
    Define some custom contracts if PyContracts is found
    """
    from contracts import new_contract

    @new_contract
    def cid_like(value):
        """
        Value is a ComponentID or a string
        """
        from glue.core import ComponentID
        return isinstance(value, (ComponentID, string_types))

    @new_contract
    def component_like(value):
        from glue.core import Component, ComponentLink
        return isinstance(value, (Component, ComponentLink,
                                  ndarray, list, Series))

    @new_contract
    def array_like(value):
        return isinstance(value, (ndarray, list))

    @new_contract
    def color(value):
        """
        A valid matplotlib color
        """
        from matplotlib.colors import colorConverter
        try:
            colorConverter.to_rgba(value)
        except ValueError:
            return False

    @new_contract
    def inst(value, *types):
        return isinstance(value, types)

    @new_contract
    def data_view(value):
        from glue.core import ComponentID

        if value is None:
            return
        if isinstance(value, ComponentID):
            return
        try:
            if not isinstance(value[0], ComponentID):
                return False
            s_[value[1:]]
        except:
            return False

    @new_contract
    def array_view(value):
        try:
            s_[value]
        except:
            return False

    @new_contract
    def callable(value):
        return hasattr(value, '__call__')

try:
    from contracts import contract, ContractsMeta

    if not enable_contracts():
        from contracts import disable_all
        disable_all()

    _build_custom_contracts()

except ImportError:
    # no-op interface if PyContracts isn't installed

    def contract(*args, **kwargs):
        if args:  # called as @contract
            return args[0]
        else:   # called as @contract(x='int', ...)
            return lambda func: func

    ContractsMeta = type
