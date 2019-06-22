Installing the latest developer version
=======================================

.. note:: The latest developer version is not guaranteed to work correctly
          or be functional at all, so use with care!

With conda
----------

If you use Anaconda/conda to install glue normally, we provide nightly builds of
the conda packages for the latest developer versions. Unless you want to
actively develop glue, this is the best way to try out the latest developer
version. We recommend that you install the developer version into a conda
environment in case you also want to be able to have the stable version of glue
in your normal environment (we'll show how to do this in the next few steps).

To create an environment (which only needs to be done the first time), type::

    conda create -n glueviz-dev python

Then switch to the ``glueviz-dev`` environment::

    source activate glueviz-dev

and install the latest nightly builds of the glue packages with::

    conda install -c glueviz -c glueviz/label/dev glueviz

You should normally see long version numbers for the glue-* packages that get
installed::

    $ conda install -c glueviz -c glueviz/label/dev glueviz
    Fetching package metadata .............
    Solving package specifications: .

    Package plan for installation in environment /Users/tom/miniconda3/envs/glue-dev:

    The following NEW packages will be INSTALLED:

        glue-core:          0.11.0.dev20170705102151.3ea9531-py36_0 glueviz/label/dev
        glue-vispy-viewers: 0.8.dev20170602171439.7533769-py36_0    glueviz/label/dev
        glueviz:            0.11.0.dev20170705211525.3af839b-0      glueviz/label/dev
        pyopengl:           3.1.1a1-np113py36_0

    Proceed ([y]/n)? y

To update to a more recent version of the developer packages, use the same
command. If you want to switch back to the original environment you were in, you
can type::

    source activate <environment_name>

where ``<environment_name>>`` might be e.g. ``root`` or ``glueviz-env``
depending on how you chose to set up your stable glue environment.

From source (if you use conda)
------------------------------

If you use conda but want to install the latest developer version from the git
repository (for example if you want to work on the code) then the easiest way to
get all the dependencies installed is to first install the stable version, which
will pull in all the dependencies, then to remove it and install the developer
version::

    conda install -c glueviz glueviz
    conda remove glueviz

    git clone https://github.com/glue-viz/glue.git
    cd glue
    pip install .
    cd ..

You can also use ``python setup.py develop`` instead of ``pip install .`` if you
want changes made in the local repository to be reflected immediately in the
installed version. Note that you can do all this in an environment as described
in `With conda`_ if you want to have the stable version of glue in a separate
environment.

The same instructions apply to other glue packages, for example the plugin with
the 3D viewers::

    conda install -c glueviz glue-vispy-viewers
    conda remove glue-vispy-viewers

    git clone https://github.com/glue-viz/glue-vispy-viewers.git
    cd glue-vispy-viewers
    pip install .
    cd ..

If you want to uninstall the developer versions and install the stable versions
again, you can uninstall the developer versions with::

    pip uninstall glueviz glue-vispy-viewers

then install the stable versions with conda as usual.

From source (if you don't use conda)
------------------------------------

If you don't use conda, but use ``pip`` instead, then you can install the latest
version of the glue core package using::

    git clone https://github.com/glue-viz/glue.git
    cd glue
    pip install -e .
    cd ..
