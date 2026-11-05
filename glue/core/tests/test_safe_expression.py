import math
import types

import pytest
import numpy as np
from numpy.testing import assert_equal

from ..safe_expression import (UnsafeExpressionError, build_namespace,
                               evaluate_expression, is_computational_callable)


def evaluate(expression, env=None, **values):
    namespace = build_namespace(env)
    namespace.update(values)
    return evaluate_expression(expression, namespace)


@pytest.mark.parametrize(('expression', 'expected'), [
    ('1 + 2 * 3 - 4 / 2', 5.0),
    ('2 ** 10 // 3 % 7', 2 ** 10 // 3 % 7),
    ('-x + abs(-2)', -1),
    ('max(x, 100)', 100),
    ('np.sqrt(16) + numpy.log10(100) + math.floor(2.5)', 8.0),
    ('1 < x < 5', True),
    ('1 < x > 5', False),
    ('x if x > 2 else 0', 3),
    ('True and x', 3),
    ('0 or x', 3),
    ('not x', False),
    ('[1, 2, 3][1:][0]', 2),
    ('np.linalg.norm([3, 4])', 5.0),
    ('np.pi', np.pi),
])
def test_scalar_expressions(expression, expected):
    assert evaluate(expression, x=3) == expected


def test_array_expressions():
    x = np.array([1., 4., 9.])
    assert_equal(evaluate('np.sqrt(x) + 1', x=x), [2, 3, 4])
    assert_equal(evaluate('(x > 2) & (x < 5)', x=x), [False, True, False])
    assert_equal(evaluate('x[x > 2]', x=x), [4, 9])
    assert_equal(evaluate('x.astype(int).sum()', x=x), 14)
    assert_equal(evaluate('np.nansum(x, axis=0)', x=x), 14)
    assert_equal(evaluate("x.astype('float32').dtype == np.float32", x=x), True)
    assert_equal(evaluate('x.T.shape', x=x), (3,))


def test_undefined_name():
    with pytest.raises(NameError, match="name 'nump' is not defined"):
        evaluate('nump.log10(1)')


@pytest.mark.parametrize('expression', [
    "__import__('math')",
    'math.__dict__',
    "np.ones(3).__class__",
    'eval',
    'open',
    'getattr',
    'type',
])
def test_disallowed_names_and_attributes(expression):
    with pytest.raises((UnsafeExpressionError, NameError)):
        evaluate(expression)


@pytest.mark.parametrize('expression', [
    "np.load('file.npy')",
    "np.save('file.npy', x)",
    "np.savetxt('file.txt', x)",
    "np.fromfile('file.dat')",
    "np.memmap('file.dat')",
    "np.vectorize(abs)",
    "np.testing",
    "np.ctypeslib",
    "np.lib",
    "x.tofile('file.dat')",
    "x.dump('file.pkl')",
    "x.ctypes.data_as",
])
def test_disallowed_numpy(expression, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UnsafeExpressionError):
        evaluate(expression, x=np.arange(3))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('expression', [
    'lambda: 1',
    '[i for i in range(3)]',
    '(y := 1)',
    "f'{x}'",
    "'{0}'.format(x)",
    'abs(*[1])',
    'max(**{})',
])
def test_disallowed_syntax(expression):
    with pytest.raises((UnsafeExpressionError, SyntaxError)):
        evaluate(expression, x=1)


def make_env(**names):
    env = types.ModuleType('config')
    for name, value in names.items():
        setattr(env, name, value)
    return env


def test_config_names():

    def triple(x):
        return 3 * x

    triple.__module__ = 'config'

    import os
    import subprocess
    import glue.config

    env = make_env(triple=triple, scale=2.5, os=os, system=os.system,
                   run=subprocess.run, glue_config=glue.config,
                   data_factory=glue.config.data_factory, _private=1)

    # Functions and values defined in config.py can be used
    assert evaluate('triple(scale)', env=env) == 7.5

    # but standard library and glue modules and functions imported into
    # config.py are not exposed, nor are private names
    for name in ['os', 'system', 'run', 'glue_config', 'data_factory', '_private']:
        with pytest.raises((NameError, UnsafeExpressionError)):
            evaluate(f'{name}', env=env)


def test_config_module_attributes():

    # Third-party modules imported in config.py can be used, but attributes
    # that lead to standard library modules or functions cannot.

    import os

    package = types.ModuleType('somepackage')
    package.compute = lambda x: x + 1
    package.compute.__module__ = 'somepackage'
    package.os = os
    package.system = os.system

    env = make_env(somepackage=package)

    assert evaluate('somepackage.compute(1)', env=env) == 2

    with pytest.raises(UnsafeExpressionError):
        evaluate('somepackage.os', env=env)

    with pytest.raises(UnsafeExpressionError):
        evaluate('somepackage.system', env=env)


def test_is_computational_callable():
    import operator
    import os
    assert is_computational_callable(np.sqrt)
    assert is_computational_callable(np.nansum)
    assert is_computational_callable(operator.add)
    assert not is_computational_callable(np.load)
    assert not is_computational_callable(os.system)
    assert not is_computational_callable(math.sqrt)
