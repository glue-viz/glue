class Translator(object):
    """
    An object to translate subsets between data.

    A translator object is created with references to one subset from
    each of two or more data objects. It subscribes to the hub, to
    receive messages when any of these subsets are modified. It then
    alters the subset in all of the other data objects to logically
    match the original change.

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

