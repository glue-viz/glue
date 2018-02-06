__all__ = ['code', 'serialize_options']


class code(str):
    pass


def serialize_options(options):
    result = []
    for key, value in options.items():
        if isinstance(value, code):
            result.append(key + '=' + value)
        else:
            result.append(key + '=' + repr(value))
    return ', '.join(result)
