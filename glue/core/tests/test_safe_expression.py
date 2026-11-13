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

    class Scale:
        factor = 2

        def __call__(self, x):
            return self.factor * x

    Scale.__module__ = 'config'

    import os
    import subprocess
    import glue.config

    def helper(x):
        return x

    helper.__module__ = 'somepackage.helpers'

    env = make_env(triple=triple, scale=2.5, Scale=Scale, os=os, system=os.system,
                   run=subprocess.run, glue_config=glue.config,
                   data_factory=glue.config.data_factory, helper=helper,
                   _private=1)

    # Functions, classes and values defined in config.py can be used
    assert evaluate('triple(scale)', env=env) == 7.5
    assert evaluate('Scale()(scale)', env=env) == 5

    # but modules and functions imported into config.py are not exposed,
    # whether they are from the standard library, glue, or other packages,
    # and nor are private names
    for name in ['os', 'system', 'run', 'glue_config', 'data_factory', 'helper', '_private']:
        with pytest.raises((NameError, UnsafeExpressionError)):
            evaluate(f'{name}', env=env)


def test_config_imported_modules():

    # Modules imported in config.py are not exposed, since they can contain
    # functions that read and write files or run code (such as
    # pandas.read_pickle), with the exception of numpy and math which are
    # always available in restricted form.

    package = types.ModuleType('somepackage')
    package.compute = lambda x: x + 1
    package.compute.__module__ = 'somepackage'

    env = make_env(somepackage=package, sp=package, xp=np, m=math)

    for name in ['somepackage', 'sp', 'xp', 'm']:
        with pytest.raises(NameError):
            evaluate(f'{name}', env=env)

    assert evaluate('np.sqrt(4) + math.sqrt(4)', env=env) == 4


def test_config_numpy_names():

    # Computational numpy functions and ufuncs imported by name into
    # config.py can be used, including ufuncs from other packages, but other
    # numpy functions cannot.

    custom_ufunc = np.frompyfunc(lambda x: x + 1, 1, 1)

    env = make_env(sqrt=np.sqrt, nansum=np.nansum, load=np.load,
                   fromfile=np.fromfile, vectorize=np.vectorize,
                   custom_ufunc=custom_ufunc)

    assert evaluate('sqrt(nansum([8, 8]))', env=env) == 4
    assert evaluate('custom_ufunc(1)', env=env) == 2

    for name in ['load', 'fromfile', 'vectorize']:
        with pytest.raises(NameError):
            evaluate(f'{name}', env=env)


def test_config_cannot_replace_builtin_names():

    # The names that are always available cannot be replaced by config.py

    def fake(*args):
        return 'replaced'

    fake.__module__ = 'config'

    env = make_env(abs=fake, np=1, math=2)

    assert evaluate('abs(-1) + np.abs(-1) + math.fabs(-1)', env=env) == 3


class Simple:
    value = 1

    def method(self):
        return 2


@pytest.mark.parametrize('expression', [
    'obj.value',
    'obj.method()',
    'obj.method',
    '(1, 2).count(1)',
    '[1, 2].append',
    'np.typecodes.get',
    'int.mro',
    'float.fromhex',
    'abs.mro',
    'np.random.default_rng().normal',
    'np.random.default_rng(0).bit_generator',
    'x.dtype.type',
    'x.dtype.newbyteorder',
    'np.add.at',
    '(1).to_bytes',
    '(1.5).hex',
])
def test_disallowed_attribute_owners(expression):
    with pytest.raises(UnsafeExpressionError):
        evaluate(expression, obj=Simple(), x=np.arange(3))


@pytest.mark.parametrize(('expression', 'expected'), [
    ('np.add.reduce(x)', 3),
    ('np.multiply.outer(x, x).shape', (3, 3)),
    ('np.add.nin', 2),
    ('x.dtype.kind', 'i'),
    ('x.dtype.itemsize', 8),
    ('(1 + 2j).real', 1),
    ('(1 + 2j).conjugate()', 1 - 2j),
    ('(2.0).is_integer()', True),
    ('x.max().real', 2),
])
def test_allowed_attribute_owners(expression, expected):
    assert evaluate(expression, x=np.arange(3, dtype=np.int64)) == expected


@pytest.mark.parametrize('expression', [
    'np.random.test',
    'np.linalg.test',
    'np.fft.test',
    "np.random.test(extra_argv=['.'])",
    'np.test',
])
def test_disallowed_numpy_test_runners(expression):
    with pytest.raises(UnsafeExpressionError):
        evaluate(expression)


def test_is_computational_callable():
    import operator
    import os
    assert is_computational_callable(np.sqrt)
    assert is_computational_callable(np.nansum)
    assert is_computational_callable(operator.add)
    assert not is_computational_callable(np.load)
    assert not is_computational_callable(os.system)
    assert not is_computational_callable(math.sqrt)
