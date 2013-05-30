import sys


def main():
    import glue
    try:
        import pytest
    except ImportError:
        raise ImportError("Glue testing requires pytest")

    return pytest.main(glue.__path__[0])

if __name__ == "__main__":
    sys.exit(main())
