import re
import random
import warnings

import numpy as np

from glue.core.component_link import ComponentLink
from glue.core.subset import Subset, SubsetState
from glue.core.data import ComponentID
from glue.core.safe_expression import (UnsafeExpressionError, build_namespace,
                                       evaluate_expression)


# The following expression matches substrings surrounded by curly brackets
# with a component name inside. The component name can be composed of any
# character that is not curly brackets (i.e. [^\{\}]) and has to start and
# end with a character that is not a curly bracket or a space. The component
# name can be surrounded by spaces, e.g. '{ a }'
TAG_RE = re.compile(r'\{\s*(?P<tag>[^\s\{\}]+([^\{\}]*[^\s\{\}]+)?)\s*\}')

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


class InvalidTagError(ValueError):
    def __init__(self, tag, references):
        msg = ("Tag %s not in reference mapping: %s" %
               (tag, sorted(references.keys())))
        self.tag = tag
        self.references = references
        super(InvalidTagError, self).__init__(msg)


def _validate(cmd, references):
    """ Make sure all references in the command are in the reference mapping

    Raises
    ------
    TypeError, if a tag is missing from references
    """

    replacements = {}
    references_new = {}
    for match in TAG_RE.finditer(cmd):
        tag = match.group('tag')
        if tag not in references:
            raise InvalidTagError(tag, references)
        full_tag = match.string[slice(*match.span())]
        replacements[full_tag] = '{' + references[tag].uuid + '}'
        references_new[references[tag].uuid] = references[tag]

    cmd_new = cmd
    for before, after in replacements.items():
        cmd_new = cmd_new.replace(before, after)

    return cmd_new, references_new


class ParsedCommand(object):

    """ Class to manage commands that define new components and subsets """

    def __init__(self, cmd, references):
        """ Create a new parsed command object

        Parameters
        ----------
        cmd : str. A template command. Can only reference ComponentID objects
        references : mapping from command templates to substitution objects
        """
        self._cmd, self._references = _validate(cmd, references)

        # Whether to check every name, attribute access and function call
        # in the command when evaluating it. This is always the case unless
        # the user has explicitly chosen to trust this command.
        self.restricted = True

        # For commands loaded from session files, this is set to the reason
        # why the command is not evaluated, if this is the case.
        self.disabled_reason = None

    def render(self, mapping=None):
        def sub_func(match):
            tag = match.group('tag')
            if mapping is None:
                label = self._references[tag].label
            else:
                label = mapping[self._references[tag]]
            return '{' + label + '}'
        return TAG_RE.sub(sub_func, self._cmd)

    def ensure_only_component_references(self):
        _ensure_only_component_references(self._cmd, self._references)

    @property
    def reference_list(self):
        return _reference_list(self._cmd, self._references)

    def _evaluate(self, values):
        """
        Evaluate the command, given a function that returns the value to use
        for each reference.
        """

        from glue import env

        namespace = build_namespace(env, restricted=self.restricted)
        names = {}

        def sub_func(match):
            tag = match.group('tag')
            if tag not in names:
                names[tag] = '_glue_ref_%i' % len(names)
                namespace[names[tag]] = values(self._references[tag])
            return names[tag]

        return evaluate_expression(TAG_RE.sub(sub_func, self._cmd), namespace,
                                   restricted=self.restricted)

    def evaluate(self, data, view=None):

        if self.disabled_reason is not None:
            raise UnsafeExpressionError(self.disabled_reason)

        def values(reference):
            if isinstance(reference, ComponentID):
                return data[reference, view]
            elif isinstance(reference, Subset):
                return reference.to_mask(view)
            else:
                raise TypeError("Reference %s maps to unrecognized type: %s" %
                                (reference, type(reference)))

        result = self._evaluate(values)

        if data is not None and np.isscalar(result):
            result = np.ones(data.shape) * result

        return result

    def evaluate_test(self, view=None):
        # Substitute random floating-point values for the references, to
        # quickly test that the command evaluates without errors.
        return self._evaluate(lambda reference: random.random())

    def __gluestate__(self, context):
        return dict(cmd=self._cmd,
                    references=dict((k, context.id(v))
                                    for k, v in self._references.items()))

    @classmethod
    def __setgluestate__(cls, rec, context):
        cmd = rec['cmd']
        ref = dict((k, context.object(v))
                   for k, v in rec['references'].items())
        self = cls(cmd, ref)
        if cmd in (getattr(context, 'allowed_expressions', None) or ()):
            # The user has explicitly chosen to trust this expression
            self.restricted = False
        else:
            self.disabled_reason = unsafe_expression_reason(cmd)
            if self.disabled_reason is not None:
                warnings.warn(f"The expression '{cmd}' from the session file will not "
                              f"be evaluated: {self.disabled_reason}")
        return self


def unsafe_expression_reason(cmd):
    """
    Check whether an expression can be evaluated with only the names,
    attributes and functions that are allowed by default.

    This is done by evaluating the expression in restricted mode with small
    arrays in place of the references, so nothing that is not allowed gets
    called.

    Parameters
    ----------
    cmd : str
        The expression, which can include references in curly brackets.

    Returns
    -------
    reason : str or None
        `None` if the expression only uses what is allowed by default, and
        otherwise a description of what is not allowed.
    """

    from glue import env

    namespace = build_namespace(env)
    names = {}

    def sub_func(match):
        tag = match.group('tag')
        if tag not in names:
            names[tag] = '_glue_ref_%i' % len(names)
            namespace[names[tag]] = np.ones(2)
        return names[tag]

    try:
        evaluate_expression(TAG_RE.sub(sub_func, cmd), namespace)
    except (UnsafeExpressionError, NameError, SyntaxError) as exc:
        return str(exc) or type(exc).__name__
    except Exception:
        # Any other error is due to the placeholder values used here rather
        # than to the expression using something that is not allowed.
        pass

    return None


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
            parsed.reference_list, to_, lambda: None)
        self._parsed = parsed

    def compute(self, data, view=None):
        if self._parsed.disabled_reason is not None:
            # The expression was loaded from a session file and uses
            # something that is not allowed by default, so we return an
            # array of NaN values with the right shape rather than failing.
            if self._from:
                shape = np.shape(data[self._from[0], view])
            else:
                shape = np.shape(np.empty(data.shape)[view] if view is not None
                                 else np.empty(data.shape))
            return np.full(shape, np.nan)
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
