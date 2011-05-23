class Translator(object):
    """
    An object to translate subsets between data

    This is the base class for all Translator objects, and
    doesn't implement any translation on its own. Subclasses
    should override the translate method.
    """

    def __init__(self):
        """Create a new translator object."""
        pass

    def translate(self, subset, data, *args, **kwargs):
        """
        Translate a subset for another data set

        If the translator does not know how to
        translate to the requested dataset,
        it returns None.

        Parameters
        ----------
        subset: Subset instance
            The subset to translate.
        data: Data instance
            The new dataset to translate to.
        """
        pass
