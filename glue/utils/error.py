
class GlueDeprecationWarning(UserWarning):
    """
    Deprecation warnings for glue - this inherits from UserWarning not
    DeprecationWarning, to make sure it is shown by default.
    """
