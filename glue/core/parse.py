import re

from .data import ComponentID
from .subset import Subset, SubsetState
from .component_link import ComponentLink

TAG_RE = re.compile('\{\s*(?P<tag>\S+)\s*\}')

__all__ = ['ParsedCommand', 'ParsedSubsetState']


def _ensure_only_component_references(cmd, references):
    """ Search through tag references in a command, ensure that
    they all reference ComponentIDs

    Parameters
    ----------
    cmd : string. A template command
    referenes : a mapping from tags to substitution objects

    Raises
    ------
    TypeError, if cmd does not refer only to ComponentIDs
    """
    for match in TAG_RE.finditer(cmd):
        tag = match.group('tag')
        if tag not in references or not \
                isinstance(references[tag], ComponentID):
            raise TypeError(
                "Reference to %s, which is not a ComponentID" % tag)


def _reference_list(cmd, references):
    """ Return a list of the values in the references mapping whose
    keys appear in the command

    Parameters
    ----------
    cmd : string. A template command
    references : a mapping from tags to substitution objects

    Returns
    -------
    A list of the unique values in references that appear in the command

    Examples
    --------
    >>> cmd = '{g} - {r} + {g}'
    >>> references = {'g' : g_object, 'r' : r_object, 'i' : i_object}
    >>> _reference_list(cmd, references)
    [g_object, r_object]

    Raises
    ------
    KeyError: if tags in the command aren't in the reference mapping
    """
    try:
        return list(set(references[m.group('tag')]
                        for m in TAG_RE.finditer(cmd)))
    except KeyError:
        raise KeyError("Tags from command not in reference mapping")


def _dereference(cmd, references):
    """ Dereference references in the template command, to refer
    to objects in the reference mapping

    Parameters
    ----------
    cmd : Command string
    references : mapping from template tags to objects

    Returns
    -------
    A new command, where all the tags have been subsituted as follows:
      "{tag}" -> 'data[references["tag"], __view]', if references[tag] is a ComponentID
      "{tag}" -> 'references["tag"].to_mask(__view)' if references[tag] is a Subset

    __view is a placeholder variable referencing the view
    passed to data.__getitem__ and subset.to_mask

    Raises
    ------
    TypeError, if a tag in the command maps to something other than
    a ComponentID or Subset object
    """
    def sub_func(match):
        tag = match.group('tag')
        if isinstance(references[tag], ComponentID):
            return 'data[references["%s"], __view]' % tag
        elif isinstance(references[tag], Subset):
            return 'references["%s"].to_mask(__view)' % tag
        else:
            raise TypeError("Tag %s maps to unrecognized type: %s" %
                            (tag, type(references[tag])))
    return TAG_RE.sub(sub_func, cmd)


def _validate(cmd, references):
    """ Make sure all references in the command are in the reference mapping

    Raises
    ------
    TypeError, if a tag is missing from references
    """
    for match in TAG_RE.finditer(cmd):
        tag = match.group('tag')
        if tag not in references:
            raise TypeError("Tag %s not in reference mapping: %s" %
                            (tag, references.keys()))


class ParsedCommand(object):
    """ Class to manage commands that define new components and subsets """

    def __init__(self, cmd, references):
        """ Create a new parsed command object

        Parameters
        ----------
        cmd : str. A template command. Can only reference ComponentID objects
        references : mapping from command templates to substitution objects
        """
        _validate(cmd, references)
        self._cmd = cmd
        self._references = references

    def ensure_only_component_references(self):
        _ensure_only_component_references(self._cmd, self._references)

    @property
    def reference_list(self):
        return _reference_list(self._cmd, self._references)

    def evaluate(self, data, view=None):
        from .. import env
        # pylint: disable=W0613, W0612
        references = self._references
        cmd = _dereference(self._cmd, self._references)
        scope = vars(env)
        scope['__view'] = view
        return eval(cmd, vars(env), locals())  # careful!

    def __gluestate__(self, context):
        return dict(cmd=self._cmd,
                    references=dict((k, context.id(v))
                                    for k, v in self._references.items()))

    @classmethod
    def __setgluestate__(cls, rec, context):
        cmd = rec['cmd']
        ref = dict((k, context.object(v))
                   for k, v in rec['references'].items())
        return cls(cmd, ref)


class ParsedComponentLink(ComponentLink):
    """ Class to create a new ComponentLink from a ParsedCommand object. """

    def __init__(self, to_, parsed):
        """ Create a new link

        Parameters
        ----------
        to_ : ComponentID instance to associate with the new component
        parsed : A ParsedCommand object
        """
        parsed.ensure_only_component_references()
        super(ParsedComponentLink, self).__init__(
            parsed.reference_list, to_, 'dummy')
        self._parsed = parsed

    def compute(self, data, view=None):
        return self._parsed.evaluate(data, view)

    def __gluestate__(self, context):
        return dict(parsed=context.do(self._parsed),
                    to=context.id(self.get_to_id()))

    @classmethod
    def __setgluestate__(cls, rec, context):
        return cls(context.object(rec['to']),
                   context.object(rec['parsed']))


class ParsedSubsetState(SubsetState):
    """ A SubsetState defined by a ParsedCommand object """
    def __init__(self, parsed):
        """ Create a new object

        Parameters
        ----------
        parsed : A ParsedCommand object
        """
        super(ParsedSubsetState, self).__init__()
        self._parsed = parsed

    def to_mask(self, data, view=None):
        """ Calculate the new mask by evaluating the dereferenced command """
        result = self._parsed.evaluate(data)
        if view is not None:
            result = result[view]
        return result
