"""
The descriptors in this module are meant to be added to classes, to
specify simple user-settable forms. These classes are used to automatically
construct GUIs, without having to write GUI code in the form class itself.

:class:`Option` objects are defined at the class-level. To instances of
these classes, an :class:`Option` behaves like a normal instance attribute.

See :ref:`fit_plugins` for example usage.
"""


class Option(object):

    """
    Base class for other options.

    This shouldn't be used directly
    """

    def __init__(self, default, label):
        """
        :param default: The default value for this option.
        :type default: object
        :param label: A short label for this option, to use in the GUI
        :type label: str
        """
        self.label = label
        """A UI label for the setting"""
        self.default = default
        """The default value"""

        self._name = "__%s_%i" % (type(self), id(self))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance, self._name, self.default)

    def __set__(self, instance, value):
        value = self._validate(value)
        setattr(instance, self._name, value)

    def _validate(self, value):
        return value


class IntOption(Option):

    def __init__(self, min=0, max=10, default=1, label="Integer"):
        """
        An integer-valued option

        :param min: The minimum valid value
        :type min: integer
        :param max: The maximum valid value
        :type max: integer
        :param default: The default value
        :type default: integer
        :param label: A short label for this option
        :type label: str
        """
        super(IntOption, self).__init__(default, label)

        self.min = min
        self.max = max

    def _validate(self, value):

        try:
            if value != int(value):
                raise ValueError()
            value = int(value)
        except ValueError:
            raise ValueError("%s must be an integer" % self.label)

        if value < self.min:
            raise ValueError("%s must be >= %i" % (self.label, self.min))

        if value > self.max:
            raise ValueError("%s must be <= %i" % (self.label, self.max))

        return value


class FloatOption(Option):

    def __init__(self, min=0, max=10, default=1, label="Float"):
        """
        An floating-point option

        :param min: The minimum valid value
        :type min: float
        :param max: The maximum valid value
        :type max: float
        :param default: The default value
        :type default: float
        :param label: A short label for this option
        :type label: str
        """
        super(FloatOption, self).__init__(default, label)
        self.min = min
        self.max = max

    def _validate(self, value):
        value = float(value)

        if value < self.min or value > self.max:
            raise ValueError("%s must be between %e and %e" % (self.label,
                                                               self.min, self.max))
        return value


class BoolOption(Option):

    def __init__(self, label="Bool", default=False):
        """
        A boolean-valued option

        :param default: The default True/False value
        :type default: bool

        :param label: A short label for this option
        :type label: str
        """
        super(BoolOption, self).__init__(default, label)

    def _validate(self, value):
        if value not in [True, False]:
            raise ValueError(
                "%s must be True or False: %s" % (self.label, value))

        return value
