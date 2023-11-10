# Licensed under a 3-clause BSD style license - see LICENSE.rst

from functools import wraps

import pytest

try:
    import pytest_mpl  # noqa
except ImportError:
    HAS_PYTEST_MPL = False
else:
    HAS_PYTEST_MPL = True


def visual_test(*args, **kwargs):
    """
    A decorator that defines a visual test.

    This automatically decorates tests with mpl_image_compare with common
    options used by all figure tests in glue-core.
    """

    tolerance = kwargs.pop("tolerance", 0)
    style = kwargs.pop("style", {})
    savefig_kwargs = kwargs.pop("savefig_kwargs", {})
    savefig_kwargs["metadata"] = {"Software": None}

    def decorator(test_function):
        @pytest.mark.mpl_image_compare(
            tolerance=tolerance, style=style, savefig_kwargs=savefig_kwargs, **kwargs
        )
        @pytest.mark.skipif(
            not HAS_PYTEST_MPL, reason="pytest-mpl is required for the figure tests"
        )
        @wraps(test_function)
        def test_wrapper(*args, **kwargs):
            return test_function(*args, **kwargs)

        return test_wrapper

    # If the decorator was used without any arguments, the only positional
    # argument will be the test to decorate so we do the following:
    if len(args) == 1:
        return decorator(*args)

    return decorator
