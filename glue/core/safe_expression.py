"""
Restricted evaluation of the arithmetic expressions used for derived
components and subsets (see :mod:`glue.core.parse`).

These expressions are stored in session files, so they cannot be passed to
:func:`eval` since a session file could then run arbitrary code when it is
opened. Instead, the expression is parsed with :mod:`ast` and the tree is
interpreted here, with every name lookup, attribute access, and call checked
against what is explicitly made available:

* the component and subset references in the expression;
* ``np``/``numpy`` and ``math``, restricted to computational functions;
* a few pure builtins such as ``abs``, ``min`` and ``max``;
* public names defined in the user's ``config.py`` (``glue.env``), since the
  user wrote that file and chose to expose them. Modules and callables from
  the standard library or from glue itself are not exposed even if they
  appear in ``config.py``, since those are typically imported there for
  other purposes.
"""

import ast
import builtins
import math
import operator
import sys
import types

import numpy as np

__all__ = ['UnsafeExpressionError', 'build_namespace', 'evaluate_expression',
           'is_computational_callable']


class UnsafeExpressionError(ValueError):
    """
    Raised when an expression uses a construct that is not allowed.
    """


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}

_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in ['abs', 'all', 'any', 'bool', 'complex', 'divmod', 'float',
                 'int', 'len', 'max', 'min', 'pow', 'round', 'sum']
}

# Functions available from the top-level numpy namespace, in addition to
# ufuncs (which are always allowed). Anything performing I/O or able to wrap
# or import other code is deliberately left out.
_NUMPY_FUNCTIONS = {
    'abs', 'all', 'allclose', 'amax', 'amin', 'angle', 'any', 'append',
    'arange', 'argmax', 'argmin', 'argsort', 'around', 'array', 'array_equal',
    'asarray', 'atleast_1d', 'atleast_2d', 'atleast_3d', 'average', 'broadcast_to',
    'ceil', 'clip', 'column_stack', 'concatenate', 'convolve', 'copy', 'corrcoef',
    'count_nonzero', 'cov', 'cross', 'cumprod', 'cumsum', 'deg2rad', 'degrees',
    'diag', 'diff', 'digitize', 'dot', 'dstack', 'einsum', 'empty_like', 'expand_dims',
    'flip', 'fliplr', 'flipud', 'full', 'full_like', 'gradient', 'histogram',
    'hstack', 'hypot', 'interp', 'isclose', 'isin', 'linspace', 'logspace',
    'max', 'mean', 'median', 'meshgrid', 'min', 'moveaxis', 'nan_to_num',
    'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean',
    'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd',
    'nansum', 'nanvar', 'ones', 'ones_like', 'outer', 'percentile', 'prod',
    'ptp', 'quantile', 'rad2deg', 'radians', 'ravel', 'repeat', 'reshape', 'roll',
    'rot90', 'round', 'searchsorted', 'select', 'shape', 'sign', 'sinc', 'size',
    'sort', 'squeeze', 'stack', 'std', 'sum', 'swapaxes', 'take', 'tile', 'trace',
    'transpose', 'trapezoid', 'tril', 'triu', 'trunc', 'unique', 'unwrap', 'var',
    'vstack', 'where', 'zeros', 'zeros_like',
    # dtypes and scalar types
    'bool_', 'complex64', 'complex128', 'float16', 'float32', 'float64',
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
}

# Submodules of numpy whose contents are purely computational.
_NUMPY_SUBMODULES = {'numpy.linalg', 'numpy.fft', 'numpy.random'}

# Methods that can be called on arrays and numpy scalars. Methods such as
# tofile and dump are excluded since they write to disk.
_ARRAY_METHODS = {
    'all', 'any', 'argmax', 'argmin', 'argsort', 'astype', 'clip', 'conj',
    'conjugate', 'copy', 'cumprod', 'cumsum', 'diagonal', 'dot', 'flatten',
    'item', 'max', 'mean', 'min', 'nonzero', 'prod', 'ptp', 'ravel', 'repeat',
    'reshape', 'round', 'squeeze', 'std', 'sum', 'swapaxes', 'take', 'tolist',
    'trace', 'transpose', 'var',
}

# Top-level packages whose modules and callables are never exposed, even if
# they are present in the user's config.py namespace.
_BLOCKED_PACKAGES = (set(sys.stdlib_module_names) - {'math', 'cmath'}) | {'glue'}


_OPERATOR_FUNCTIONS = set(_BINARY_OPERATORS.values()) | {
    func for func in _COMPARISON_OPERATORS.values() if func.__module__ in ('operator', '_operator')
} | set(_UNARY_OPERATORS.values())


def is_computational_callable(func):
    """
    Whether ``func`` is an arithmetic or comparison function from the
    operator module, a ufunc, or one of the computational functions in the
    top-level numpy namespace.
    """
    if isinstance(func, np.ufunc):
        return True
    try:
        if func in _OPERATOR_FUNCTIONS:
            return True
    except TypeError:  # unhashable
        pass
    name = getattr(func, '__name__', None)
    return name in _NUMPY_FUNCTIONS and getattr(np, name, None) is func


def _root_package(name):
    return (name or '').split('.')[0]


def _check_numpy_value(value, name, owner):
    """
    Check a value obtained from the numpy namespace.
    """
    if isinstance(value, types.ModuleType):
        if value.__name__ == 'numpy' or value.__name__ in _NUMPY_SUBMODULES:
            return value
        raise UnsafeExpressionError(f"Access to module {value.__name__} is not allowed")
    if isinstance(value, np.ufunc):
        return value
    if owner is not None and getattr(owner, '__name__', None) in _NUMPY_SUBMODULES:
        return value
    if name in _NUMPY_FUNCTIONS or not callable(value):
        return value
    raise UnsafeExpressionError(f"numpy.{name} is not allowed in expressions")


def _check_value(value, name, owner=None):
    """
    Check whether a value obtained by name or attribute lookup can be used.
    """

    if owner is not None and isinstance(owner, (np.ndarray, np.generic)):
        if callable(value) and name not in _ARRAY_METHODS:
            raise UnsafeExpressionError(f"Array method {name} is not allowed in expressions")
        return value

    if isinstance(value, types.ModuleType):
        module_name = value.__name__
    elif callable(value):
        module_name = getattr(value, '__module__', None)
    else:
        return value

    root = _root_package(module_name)

    if root == 'numpy':
        return _check_numpy_value(value, name, owner)

    if root in _BLOCKED_PACKAGES:
        if isinstance(value, types.ModuleType):
            raise UnsafeExpressionError(f"Access to module {module_name} is not allowed")
        raise UnsafeExpressionError(f"{name} (from {module_name}) is not allowed in expressions")

    return value


def build_namespace(env=None):
    """
    Build the namespace of names available in expressions, given the user
    configuration module ``env``.
    """

    namespace = dict(_SAFE_BUILTINS)
    namespace.update(np=np, numpy=np, math=math)

    if env is not None:
        for name, value in vars(env).items():
            if name.startswith('_'):
                continue
            try:
                namespace[name] = _check_value(value, name)
            except UnsafeExpressionError:
                continue

    return namespace


class _Evaluator:

    def __init__(self, namespace):
        self.namespace = namespace

    def visit(self, node):
        method = getattr(self, 'visit_' + type(node).__name__, None)
        if method is None:
            raise UnsafeExpressionError(f"{type(node).__name__} is not allowed in expressions")
        return method(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        # Strings are allowed as plain values (e.g. dtype names), but their
        # methods (such as format) are blocked in visit_Attribute.
        return node.value

    def visit_Name(self, node):
        if node.id.startswith('_') and node.id not in self.namespace:
            raise UnsafeExpressionError(f"Name {node.id} is not allowed in expressions")
        try:
            return self.namespace[node.id]
        except KeyError:
            raise NameError(f"name '{node.id}' is not defined")

    def visit_Attribute(self, node):
        if node.attr.startswith('_'):
            raise UnsafeExpressionError(f"Attribute {node.attr} is not allowed in expressions")
        owner = self.visit(node.value)
        if isinstance(owner, (str, bytes)):
            raise UnsafeExpressionError("String attributes are not allowed in expressions")
        value = getattr(owner, node.attr)
        return _check_value(value, node.attr, owner=owner)

    def visit_Call(self, node):
        func = self.visit(node.func)
        if not callable(func):
            raise TypeError(f"{func!r} is not callable")
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise UnsafeExpressionError("Starred arguments are not allowed in expressions")
            args.append(self.visit(arg))
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise UnsafeExpressionError("** arguments are not allowed in expressions")
            kwargs[keyword.arg] = self.visit(keyword.value)
        return func(*args, **kwargs)

    def visit_BinOp(self, node):
        op = _BINARY_OPERATORS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(f"Operator {type(node.op).__name__} is not allowed")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        return _UNARY_OPERATORS[type(node.op)](self.visit(node.operand))

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            value = True
            for item in node.values:
                value = self.visit(item)
                if not value:
                    return value
            return value
        else:
            value = False
            for item in node.values:
                value = self.visit(item)
                if value:
                    return value
            return value

    def visit_Compare(self, node):
        # Chained comparisons such as a < b < c are evaluated as
        # (a < b) and (b < c), as in Python.
        left = self.visit(node.left)
        last = len(node.ops) - 1
        for index, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
            right = self.visit(comparator)
            result = _COMPARISON_OPERATORS[type(op)](left, right)
            if index < last and not result:
                return result
            left = right
        return result

    def visit_IfExp(self, node):
        return self.visit(node.body) if self.visit(node.test) else self.visit(node.orelse)

    def visit_Subscript(self, node):
        return self.visit(node.value)[self.visit(node.slice)]

    def visit_Slice(self, node):
        return slice(None if node.lower is None else self.visit(node.lower),
                     None if node.upper is None else self.visit(node.upper),
                     None if node.step is None else self.visit(node.step))

    def visit_Tuple(self, node):
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_List(self, node):
        return [self.visit(elt) for elt in node.elts]


def evaluate_expression(expression, namespace):
    """
    Evaluate an expression with only the given names available.

    Parameters
    ----------
    expression : str
        The expression to evaluate.
    namespace : dict
        Names available in the expression, typically built with
        :func:`build_namespace` and updated with the component references.
    """
    tree = ast.parse(expression.strip(), mode='eval')
    return _Evaluator(namespace).visit(tree)
