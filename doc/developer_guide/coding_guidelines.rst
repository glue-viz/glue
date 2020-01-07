Coding guidelines
=================

Glue is written entirely in Python, and we abide by the following guidelines:

* We follow many of the same guidelines as the `Astropy <https://www.astropy.org>`_ project, which you can find `here <http://docs.astropy.org/en/stable/development/codeguide.html#coding-style-conventions>`__.

* We use absolute imports for most of the code in Glue, with the exception of
  tests, which are allowed to import the classes/functions they are testing
  using relative imports. This means that if we need to move files and their
  associated tests around, the tests will still work without having to change
  the imports.

* All Qt-specific code should live in ``qt/`` sub-directories (see
  :ref:`qt_code` for more details).

* Docstrings should be written using the `numpydoc
  <https://github.com/numpy/numpydoc>`_ format, which is described in detail
  `here <http://docs.astropy.org/en/latest/development/docrules.html>`__.
