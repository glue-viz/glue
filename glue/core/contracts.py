from __future__ import absolute_import
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
import re

from numpy import ndarray

from ..config import enable_contracts
from ..external.six import string_types


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
        from . import ComponentID
        return isinstance(value, (ComponentID, string_types))

    @new_contract
    def component_like(value):
        from . import Component, ComponentLink
        return isinstance(value, (Component, ComponentLink,
                                  ndarray, list))

    @new_contract
    def array_like(value):
        return isinstance(value, (ndarray, list))

    @new_contract
    def color(value):
        return isinstance(value, string_types) and \
            re.match('#[0-9A-F]{6}', value) is not None

    @new_contract
    def inst(value, *types):
        return isinstance(value, types)


try:
    from contracts import contract

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
