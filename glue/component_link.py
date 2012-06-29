class ComponentLink(object):
    def __init__(self, comp_from, comp_to, using=None):
        """
        Inputs:
        -------
        comp_from: A collection of component IDs
        comp_to: The target component ID
        using: The translation function which maps data from comp_from
               to comp_to::

               using(data[comp_from[0]],...,data[comp_from[-1]]) = desired data

        """
        self._from = comp_from
        self._to = comp_to
        self._using = using

        if type(comp_from) is not list:
            raise TypeError("comp_from must be a list: %s" % type(comp_from))

        if using is None:
            if len(comp_from) != 1:
                raise TypeError("comp_from must have only 1 element, "
                                "or a 'using' function must be provided")

    def compute(self, data):
        """For a given data set, compute the component comp_to given
        the data associated with each comp_from and the `using`
        function

        Inputs
        -------
        data: The data set to use

        Returns
        -------
        The data associated with comp_to component
        """
        if self._using is None:
            return data[self._from[0]]

        args = [data[f] for f in self._from]
        return self._using(*args)

    def get_from_ids(self):
        return self._from

    def get_to_id(self):
        return self._to

    def get_using(self):
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