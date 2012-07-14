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
        self._from = comp_from
        self._to = comp_to
        self._using = using or identity

        if type(comp_from) is not list:
            raise TypeError("comp_from must be a list: %s" % type(comp_from))

        if using is None:
            if len(comp_from) != 1:
                raise TypeError("comp_from must have only 1 element, "
                                "or a 'using' function must be provided")

    def compute(self, data):
        """For a given data set, compute the component comp_to given
        the data associated with each comp_from and the ``using``
        function

        :param data: The data set to use

        *Returns*:

            The data associated with comp_to component

        *Raises*:

            InvalidAttribute, if the data set doesn't have all the
            ComponentIDs needed for the transformation
        """
        if self._using is None:
            return data[self._from[0]]

        args = [data[f] for f in self._from]
        return self._using(*args)

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
        if self._using is not None:
            result = "%s <- %s(%s)" % (self._to, self._using.__name__, args)
        else:
            result = "%s <- %s" % (self._to, self._from)
        return result

    def __repr__(self):
        return str(self)
