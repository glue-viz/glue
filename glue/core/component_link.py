from __future__ import absolute_import, division, print_function

import numbers
import operator

import numpy as np

from glue.external.six import add_metaclass
from glue.core.contracts import contract, ContractsMeta
from glue.core.subset import InequalitySubsetState
from glue.core.util import join_component_view
from glue.utils import unbroadcast, broadcast_to
from glue.logger import logger

__all__ = ['ComponentLink', 'BinaryComponentLink', 'CoordinateComponentLink']


def identity(x):
    return x


def null(*args):
    return None


OPSYM = {operator.add: '+', operator.sub: '-',
         operator.truediv: '/', operator.mul: '*',
         operator.pow: '**'}


@add_metaclass(ContractsMeta)
class ComponentLink(object):

    """ ComponentLinks represent transformation logic between ComponentIDs

    ComponentLinks are be used to derive one
    :class:`~glue.core.component_id.ComponentID` from another:

    Example::

       def hours_to_minutes(hours):
           return hours * 60

       d = Data(hour=[1, 2, 3])
       hour = d.id['hour']
       minute = ComponentID('minute')
       link = ComponentLink( [hour], minute, using=hours_to_minutes)

       link.compute(d)  # array([ 60, 120, 180])
       d.add_component_link(link)
       d['minute'] # array([ 60, 120, 180])
    """

    @contract(using='callable|None',
              inverse='callable|None')
    def __init__(self, comp_from, comp_to, using=None, inverse=None, inverse_component_link=None):
        """
        :param comp_from: The input ComponentIDs
        :type comp_from: list of :class:`~glue.core.component_id.ComponentID`

        :param comp_to: The target component ID
        :type comp_from: :class:`~glue.core.component_id.ComponentID`

        :pram using: The translation function which maps data from
                     comp_from to comp_to (optional)

        The using function should satisfy::

               using(data[comp_from[0]],...,data[comp_from[-1]]) = desired data

        :param inverse:
            The inverse translation function, if exists (optional)

        :raises:
           TypeError if input is invalid

        .. note ::
            Both ``inverse`` and ``using`` should accept and return
            numpy arrays

        """
        from glue.core.data import ComponentID

        self._from = comp_from
        self._to = comp_to

        if using is None:
            using = identity

        if using is identity:
            if inverse is None:
                inverse = identity
            elif inverse is identity:
                pass
            else:
                raise ValueError("Cannot specify inverse if using is identity")

        self._using = using
        self._inverse = inverse

        self.identity = self._using is identity

        if not isinstance(comp_from, list):
            raise TypeError("comp_from must be a list: %s" % type(comp_from))

        if not all(isinstance(f, ComponentID) for f in self._from):
            raise TypeError("from argument is not a list of ComponentIDs: %s" %
                            self._from)
        if not isinstance(self._to, ComponentID):
            raise TypeError("to argument is not a ComponentID: %s" %
                            type(self._to))

        if using is identity:
            if len(comp_from) != 1:
                raise TypeError("comp_from must have only 1 element, "
                                "or a 'using' function must be provided")

        if inverse_component_link is None:
            if inverse is not None:
                if len(comp_from) == 1:
                    self._inverse_component_link = ComponentLink([self._to], self._from[0],
                                                                 using=self._inverse,
                                                                 inverse=self._using,
                                                                 inverse_component_link=self)
                else:
                    raise ValueError("Can only use an inverse with one comp_from link")
            else:
                self._inverse_component_link = None
        else:
            self._inverse_component_link = inverse_component_link

    @contract(data='isinstance(Data)', view='array_view')
    def compute(self, data, view=None):
        """
        For a given data set, compute the component comp_to given the data
        associated with each comp_from and the ``using`` function

        This raises an :class:`glue.core.exceptions.IncompatibleAttribute` if the
        data set doesn't have all the ComponentIDs needed for the transformation

        Parameters
        ----------
        data : `~glue.core.data.Data`
            The data set to use
        view : `None` or `slice` or `tuple`
            Optional view (e.g. slice) through the data to use

        Returns
        -------
        result
            The data associated with comp_to component
        """

        # First we get the values of all the 'from' components.
        args = [data[join_component_view(f, view)] for f in self._from]

        # We keep track of the original shape of the arguments
        original_shape = args[0].shape
        logger.debug("shape of first argument: %s", original_shape)

        # We now unbroadcast the arrays to only compute the link with the
        # smallest number of values we can. This can help for cases where
        # the link depends only on e.g. pixel components or world coordinates
        # that themselves only depend on a subset of pixel components.
        # Unbroadcasting is the act of returning the smallest array that
        # contains all the information needed to be broadcasted back to its
        # full value
        args = [unbroadcast(arg) for arg in args]

        # We now broadcast these to the smallest common shape in case the
        # linking functions don't know how to broadcast arrays with different
        # shapes.
        args = np.broadcast_arrays(*args)

        # We call the actual linking function
        result = self._using(*args)

        # We call asarray since link functions may return Python scalars in some cases
        result = np.asarray(result)

        # In some cases, linking functions return ravelled arrays, so we
        # fix this here.
        logger.debug("shape of result: %s", result.shape)
        if result.shape != args[0].shape:
            logger.debug("ComponentLink function %s changed shape. Fixing",
                         self._using.__name__)
            result.shape = args[0].shape

        # Finally we broadcast the final result to desired shape
        result = broadcast_to(result, original_shape)

        return result

    def get_from_ids(self):
        """ The list of input ComponentIDs """
        return self._from

    @contract(old='isinstance(ComponentID)', new='isinstance(ComponentID)')
    def replace_ids(self, old, new):
        """Replace all references to an old ComponentID with references
        to new

        :parma old: ComponentID to replace
        :param new: ComponentID to replace with
        """
        for i, f in enumerate(self._from):
            if f is old:
                self._from[i] = new
        if self._to is old:
            self._to = new

    @contract(_from='list(isinstance(ComponentID))')
    def set_from_ids(self, _from):
        if len(_from) != len(self._from):
            raise ValueError("New ID list has the wrong length.")
        self._from = _from

    def get_to_id(self):
        """ The target ComponentID """
        return self._to

    def set_to_id(self, to):
        self._to = to

    def get_using(self):
        """ The transformation function """
        return self._using

    @property
    def inverse(self):
        if self._inverse is None:
            return None
        else:
            return self._inverse_component_link

    def get_inverse(self):
        """ The inverse transformation, or None """
        return self._inverse

    def __str__(self):
        args = ", ".join([t.label for t in self._from])
        if self._using is identity:
            result = "%s <-> %s" % (self._to, self._from[0])
        else:
            if self._inverse is None:
                result = "%s <- %s(%s)" % (self._to, self._using.__name__, args)
            else:
                result = "%s <-> %s(%s)" % (self._to, self._using.__name__, args)
        return result

    def to_html(self):
        args = ", ".join([t.to_html() for t in self._from])
        if self._using is identity:
            result = "%s &#8596; %s" % (self._to.to_html(), self._from[0].to_html())
        else:
            if self._inverse is None:
                result = "%s &#8592; %s(%s)" % (self._to.to_html(), self._using.__name__, args)
            else:
                result = "%s &#8596; %s(%s)" % (self._to.to_html(), self._using.__name__, args)
        return result

    def __repr__(self):
        return str(self)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __add__(self, other):
        return BinaryComponentLink(self, other, operator.add)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __radd__(self, other):
        return BinaryComponentLink(other, self, operator.add)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __sub__(self, other):
        return BinaryComponentLink(self, other, operator.sub)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __rsub__(self, other):
        return BinaryComponentLink(other, self, operator.sub)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __mul__(self, other):
        return BinaryComponentLink(self, other, operator.mul)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __rmul__(self, other):
        return BinaryComponentLink(other, self, operator.mul)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __div__(self, other):
        return BinaryComponentLink(self, other, operator.div)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __rdiv__(self, other):
        return BinaryComponentLink(other, self, operator.div)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __truediv__(self, other):
        return BinaryComponentLink(self, other, operator.truediv)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __rtruediv__(self, other):
        return BinaryComponentLink(other, self, operator.truediv)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __pow__(self, other):
        return BinaryComponentLink(self, other, operator.pow)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __rpow__(self, other):
        return BinaryComponentLink(other, self, operator.pow)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __lt__(self, other):
        return InequalitySubsetState(self, other, operator.lt)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __le__(self, other):
        return InequalitySubsetState(self, other, operator.le)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __gt__(self, other):
        return InequalitySubsetState(self, other, operator.gt)

    @contract(other='isinstance(ComponentID)|component_like|float|int')
    def __ge__(self, other):
        return InequalitySubsetState(self, other, operator.ge)


class CoordinateComponentLink(ComponentLink):

    @contract(comp_from='list(isinstance(ComponentID))',
              comp_to='isinstance(ComponentID)',
              coords='isinstance(Coordinates)',
              index=int,
              pixel2world=bool)
    def __init__(self, comp_from, comp_to, coords, index, pixel2world=True):
        self.coords = coords
        self.index = index
        self.pixel2world = pixel2world

        # Some coords don't need all pixel coords
        # to compute a given world coord, and vice versa
        # (e.g., spectral data cubes)
        self.ndim = len(comp_from)
        self.from_needed = coords.dependent_axes(index)
        self._from_all = comp_from

        comp_from = [comp_from[i] for i in self.from_needed]
        super(CoordinateComponentLink, self).__init__(
            comp_from, comp_to, self.using)

    def using(self, *args):

        attr = 'pixel2world_single_axis' if self.pixel2world else 'world2pixel_single_axis'
        func = getattr(self.coords, attr)

        # NOTE: in the past, we set any non-specified arguemnts to 0 for the
        # input coordinates, but this caused issues because in astropy.wcs
        # if one specifies e.g. (0, 0, 3000.) for (ra, dec, velocity), and if
        # (0, 0) for RA/Dec would return (nan, nan) normally, the velocity
        # is also NaN even though it is decoupled from the other coordinates.
        default = self.coords.default_world_coords(self.ndim)

        args2 = [None] * self.ndim
        for f, a in zip(self.from_needed, args):
            args2[f] = a
        for i in range(self.ndim):
            if args2[i] is None:
                args2[i] = np.ones_like(args[0]) * default[self.ndim - 1 - i]
        args2 = tuple(args2)

        return func(*args2[::-1], axis=self.ndim - 1 - self.index)

    def __str__(self):
        rep = 'pix2world' if self.pixel2world else 'world2pix'
        sup = super(CoordinateComponentLink, self).__str__()
        return sup.replace('using', rep)


class BinaryComponentLink(ComponentLink):

    """
    A ComponentLink that combines two inputs with a binary function

    :param left: The first input argument.
                 ComponentID, ComponentLink, or number

    :param right: The second input argument.
                  ComponentID, ComponentLink, or number

    :param op: A function with two inputs that works on numpy arrays

    The CompoentLink represents the logic of applying `op` to the
    data associated with the inputs `left` and `right`.
    """

    def __init__(self, left, right, op):
        from glue.core.data import ComponentID

        self._left = left
        self._right = right
        self._op = op

        from_ = []
        if isinstance(left, ComponentID):
            from_.append(left)
        elif isinstance(left, ComponentLink):
            from_.extend(left.get_from_ids())
        elif not isinstance(left, numbers.Number):
            raise TypeError("Cannot create BinaryComponentLink using %s" %
                            left)

        if isinstance(right, ComponentID):
            from_.append(right)
        elif isinstance(right, ComponentLink):
            from_.extend(right.get_from_ids())
        elif not isinstance(right, numbers.Number):
            raise TypeError("Cannot create BinaryComponentLink using %s" %
                            right)

        to = ComponentID("")
        super(BinaryComponentLink, self).__init__(from_, to, null)

    def replace_ids(self, old, new):
        super(BinaryComponentLink, self).replace_ids(old, new)
        if self._left is old:
            self._left = new
        elif isinstance(self._left, ComponentLink):
            self._left.replace_ids(old, new)
        if self._right is old:
            self._right = new
        elif isinstance(self._right, ComponentLink):
            self._right.replace_ids(old, new)

    def compute(self, data, view=None):

        left = self._left
        right = self._right

        if not isinstance(self._left, numbers.Number):
            left = data[self._left, view]
        if not isinstance(self._right, numbers.Number):
            right = data[self._right, view]

        # As described in more detail in ComponentLink.compute, we can
        # 'unbroadcast' the arrays to ensure a minimal operation

        original_shape = None

        if isinstance(left, np.ndarray):
            original_shape = left.shape
            left = unbroadcast(left)

        if isinstance(right, np.ndarray):
            original_shape = right.shape
            right = unbroadcast(right)

        if original_shape is not None:
            left, right = np.broadcast_arrays(left, right)

        result = self._op(left, right)

        if original_shape is None:
            return result
        else:
            return broadcast_to(result, original_shape)

    def __gluestate__(self, context):
        left = context.id(self._left)
        right = context.id(self._right)
        operator = context.do(self._op)
        return dict(left=left, right=right, operator=operator)

    @classmethod
    def __setgluestate__(cls, rec, context):
        left = context.object(rec['left'])
        right = context.object(rec['right'])
        operator = context.object(rec['operator'])
        return cls(left, right, operator)

    def __str__(self):
        sym = OPSYM.get(self._op, self._op.__name__)
        return '(%s %s %s)' % (self._left, sym, self._right)

    def __repr__(self):
        return "<BinaryComponentLink: %s>" % self
