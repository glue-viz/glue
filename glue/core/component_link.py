import logging
import operator

from .util import join_component_view
from .subset import InequalitySubsetState

__all__ = ['ComponentLink']


def identity(x):
    return x


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
    def __init__(self, comp_from, comp_to, using=None):
        """
        :param comp_from: The input ComponentIDs
        :type comp_from: list of :class:`~glue.core.data.ComponentID`

        :param comp_to: The target component ID
        :type comp_from: :class:`~glue.core.data.ComponentID`

        :pram using: The translation function which maps data from
                     comp_from to comp_to
        :type using: callable

        The using function should satisfy::

               using(data[comp_from[0]],...,data[comp_from[-1]]) = desired data

        *Raises*:

           TypeError if input is invalid
        """
        from .data import ComponentID

        self._from = comp_from
        self._to = comp_to
        self._using = using or identity

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
        if self._using is None:
            return data[join_component_view(self._from[0], view)]

        args = [data[join_component_view(f, view)] for f in self._from]
        logging.debug("shape of first argument: %s", args[0].shape)
        result = self._using(*args)
        logging.debug("shape of result: %s", result.shape)
        if result.shape != args[0].shape:
            logging.warn("ComponentLink function %s changed shape. Fixing",
                         self._using.__name__)
            result.shape = args[0].shape
        return result

    def get_from_ids(self):
        """ The list of input ComponentIDs """
        return self._from

    def get_to_id(self):
        """ The target ComponentID """
        return self._to

    def get_using(self):
        """ The transformation function """
        return self._using

    def __str__(self):
        args = ", ".join([t.label for t in self._from])
        if self._using is not identity:
            result = "%s <- %s(%s)" % (self._to, self._using.__name__, args)
        else:
            result = "%s <- %s" % (self._to, self._from)
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
        from_ = []

        lid = isinstance(left, ComponentID)
        llink = isinstance(left, ComponentLink)
        lnumber = operator.isNumberType(left)
        rid = isinstance(right, ComponentID)
        rlink = isinstance(right, ComponentLink)
        rnumber = operator.isNumberType(right)

        if rnumber and lnumber:
            raise TypeError("Cannot create BinaryComponentLink from inputs: "
                            "%s %s" % (left, right))

        if lid:
            from_.append(left)
        elif llink:
            from_.extend(left.get_from_ids())

        if rid:
            from_.append(right)
        elif rlink:
            from_.extend(right.get_from_ids())

        if lid and rid:
            using = op
        elif lid and rlink:
            using = lambda *args: op(args[0], right.get_using()(*args[1:]))
        elif lid and rnumber:
            using = lambda x: op(x, right)
        elif llink and rid:
            using = lambda *args: op(left.get_using()(*args[:-1]), args[-1])
        elif llink and rlink:
            def compute(*args):
                n_l = len(left.get_from_ids())
                return op(left.get_using()(*args[:n_l]),
                          right.get_using()(*args[n_l:]))
            using = compute
        elif llink and rnumber:
            using = lambda *args: op(left.get_using()(*args), right)
        elif lnumber and rid:
            using = lambda x: op(left, x)
        elif lnumber and rlink:
            using = lambda *args: op(left, right.get_using()(*args))

        to = ComponentID('')
        super(BinaryComponentLink, self).__init__(from_, to, using)
