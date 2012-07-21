from .decorators import singleton
from collections import defaultdict
from itertools import count

def disambiguate(label, taken):
    suffix = "_%2.2i"
    label = str(label)
    for i in count(1):
        candidate = label + (suffix % i)
        if candidate not in taken:
            return candidate

@singleton
class Registry(object):
    """ Stores labels for classes of objects. Ensures uniqueness

    The registry ensures that labels for objects of the same "group"
    are unique, and disambiguates as necessary. By default,
    objects types are used to group, but anything can be used as a group

    Registry is a singleton, and thus all instances of Registry
    share the same information

    Usage:

        >>> r = Registry()
        >>> x, y, z = 3, 4, 5
        >>> w = list()
        >>> r.register(x, 'Label')
        'Label'
        >>> r.register(y, 'Label')  # duplicate label disambiguated
        'Label_01'
        >>> r.register(w, 'Label')  # uniqueness only enforced within groups
        'Label'
        >>> r.register(z, 'Label', group=int) # put z in integer registry
        'Label_02'
    """
    def __init__(self):
        self._registry = defaultdict(dict)

    def register(self, obj, label, group=None):
        """ Register label with object (possibly disamgiguating)

        :param obj: The object to label
        :param label: The desired label
        :param group: (optional) use the registry for group (default=type(obj))

        :rtype: str

        *Returns*
        The disambiguated label
        """
        group = group or type(obj)

        reg = self._registry[group]

        has_obj = obj in reg
        has_label = label in reg.values()
        label_is_obj = has_label and has_obj and reg[obj] == label

        if has_label and (not label_is_obj):
            values = set(reg.values())
            if has_obj:
                values.remove(reg[obj])
            label = disambiguate(label, values)

        reg[obj] = label
        return label

    def unregister(self, obj, group=None):
        group = group or type(obj)
        reg = self._registry[group]
        if obj in reg:
            reg.pop(obj)

    def clear(self):
        """ Reset registry, clearing all stored values """
        self._registry = defaultdict(dict)
