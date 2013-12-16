import logging
import operator

from .util import join_component_view
from .subset import InequalitySubsetState

__all__ = ['ComponentLink', 'BinaryComponentLink']


def identity(x):
    return x

OPSYM = {operator.add: '+', operator.sub: '-',
         operator.div: '/', operator.mul: '*',
         operator.pow: '**'}


class ComponentLink(object):

    """ ComponentLinks represent transformation logic between ComponentIDs

    ComponentLinks are used by Glue to derive information not stored
    directly in a data set. For example, ComponentLinks can be used
    to convert between coordinate systesms, so that regions of interest
    defined in one coordinate system can be propagated to constraints
    in the other.

    Once a ComponentLink has been created, ``link.compute(data)`` will
    return the derived information for a given data object

    Example::

       def hours_to_minutes(degrees):
           hours * 60

       hour = ComponentID()
       minute = ComponentID()
       link = ComponentLinke( [hour], minute, using=hours_to_minutes)

    Now, if a data set has information stored with the hour componentID,
    ``link.compute(data)`` will convert that information to minutes
    """

    def __init__(self, comp_from, comp_to, using=None, inverse=None):
        """
        :param comp_from: The input ComponentIDs
        :type comp_from: list of :class:`~glue.core.data.ComponentID`

        :param comp_to: The target component ID
        :type comp_from: :class:`~glue.core.data.ComponentID`

        :pram using: The translation function which maps data from
                     comp_from to comp_to
        :type using: callable, or None

        The using function should satisfy::

               using(data[comp_from[0]],...,data[comp_from[-1]]) = desired data

        :param inverse: The inverse translation function, if exists
        :type inverse: callable, or None

        *Raises*:

           TypeError if input is invalid
        """
        from .data import ComponentID

        self._from = comp_from
        self._to = comp_to
        self._using = using or identity
        self._inverse = inverse

        self.hidden = using is None or using is identity  # show in widgets?
        self.identity = self.hidden  # identity link?
        self.hide_from_editor = False

        if type(comp_from) is not list:
            raise TypeError("comp_from must be a list: %s" % type(comp_from))

        if not all(isinstance(f, ComponentID) for f in self._from):
            raise TypeError("from argument is not a list of ComponentIDs")
        if not isinstance(self._to, ComponentID):
            raise TypeError("to argument is not a ComponentID")

        if using is None:
            if len(comp_from) != 1:
                raise TypeError("comp_from must have only 1 element, "
                                "or a 'using' function must be provided")

    def compute(self, data, view=None):
        """For a given data set, compute the component comp_to given
        the data associated with each comp_from and the ``using``
        function

        :param data: The data set to use
        :param view: Optional view (e.g. slice) through the data to use


        *Returns*:

            The data associated with comp_to component

        *Raises*:

            InvalidAttribute, if the data set doesn't have all the
            ComponentIDs needed for the transformation
        """
        logger = logging.getLogger(__name__)
        args = [data[join_component_view(f, view)] for f in self._from]
        logger.debug("shape of first argument: %s", args[0].shape)
        result = self._using(*args)
        logger.debug("shape of result: %s", result.shape)
        if result.shape != args[0].shape:
            logger.warn("ComponentLink function %s changed shape. Fixing",
                        self._using.__name__)
            result.shape = args[0].shape
        return result

    def get_from_ids(self):
        """ The list of input ComponentIDs """
        return self._from

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

    def get_inverse(self):
        """ The inverse transformation, or None """
        return self._inverse

    def __str__(self):
        args = ", ".join([t.label for t in self._from])
        if self._using is not identity:
            result = "%s <- %s(%s)" % (self._to, self._using.__name__, args)
        else:
            result = "%s <-> %s" % (self._to, self._from)
        return result

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return BinaryComponentLink(self, other, operator.add)

    def __radd__(self, other):
        return BinaryComponentLink(other, self, operator.add)

    def __sub__(self, other):
        return BinaryComponentLink(self, other, operator.sub)

    def __rsub__(self, other):
        return BinaryComponentLink(other, self, operator.sub)

    def __mul__(self, other):
        return BinaryComponentLink(self, other, operator.mul)

    def __rmul__(self, other):
        return BinaryComponentLink(other, self, operator.mul)

    def __div__(self, other):
        return BinaryComponentLink(self, other, operator.div)

    def __rdiv__(self, other):
        return BinaryComponentLink(other, self, operator.div)

    def __pow__(self, other):
        return BinaryComponentLink(self, other, operator.pow)

    def __rpow__(self, other):
        return BinaryComponentLink(other, self, operator.pow)

    def __lt__(self, other):
        return InequalitySubsetState(self, other, operator.lt)

    def __le__(self, other):
        return InequalitySubsetState(self, other, operator.le)

    def __gt__(self, other):
        return InequalitySubsetState(self, other, operator.gt)

    def __ge__(self, other):
        return InequalitySubsetState(self, other, operator.ge)


class CoordinateComponentLink(ComponentLink):

    def __init__(self, comp_from, comp_to, coords, index, pixel2world=True):
        self.coords = coords
        self.index = index
        self.pixel2world = pixel2world
        super(CoordinateComponentLink, self).__init__(
            comp_from, comp_to, self.using)

    def using(self, *args):
        attr = 'pixel2world' if self.pixel2world else 'world2pixel'
        func = getattr(self.coords, attr)
        return func(*args[::-1])[::-1][self.index]


class BinaryComponentLink(ComponentLink):

    """
    A ComponentLink that combines two inputs with a binary function

    :param left: The first input argument
    :type left: ComponentID, ComponentLink, or number

    :param right: The second input argument
    :type right: ComponentID, ComponentLink, or number

    :param op: A function with two inputs that works on numpy arrays

    The CompoentLink represents the logic of applying `op` to the
    data associated with the inputs `left` and `right`.
    """

    def __init__(self, left, right, op):
        from .data import ComponentID

        self._left = left
        self._right = right
        self._op = op

        from_ = []
        if isinstance(left, ComponentID):
            from_.append(left)
        elif isinstance(left, ComponentLink):
            from_.extend(left.get_from_ids())
        elif not operator.isNumberType(left):
            raise TypeError("Cannot create BinaryComponentLink using %s" %
                            left)

        if isinstance(right, ComponentID):
            from_.append(right)
        elif isinstance(right, ComponentLink):
            from_.extend(right.get_from_ids())
        elif not operator.isNumberType(right):
            raise TypeError("Cannot create BinaryComponentLink using %s" %
                            right)

        to = ComponentID("")
        null = lambda *args: None
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
        l = self._left
        r = self._right
        if not operator.isNumberType(self._left):
            l = data[self._left, view]
        if not operator.isNumberType(self._right):
            r = data[self._right, view]
        return self._op(l, r)

    def __str__(self):
        sym = OPSYM.get(self._op, self._op.__name__)
        return '(%s %s %s)' % (self._left, sym, self._right)

    def __repr__(self):
        return "<BinaryComponentLink: %s>" % self
