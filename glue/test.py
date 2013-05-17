if __name__ == "__main__":
    import glue
    import sys
    try:
        import pytest
    except ImportError:
        raise ImportError("Glue testing requires pytest")
    sys.exit(pytest.main(glue.__path__[0]))
