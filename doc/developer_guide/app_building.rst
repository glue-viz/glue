How the MacOS X app is built
============================

While we recommend that you use Glue by installing the ``glueviz`` package in
Conda, we also make available an all-in-one app for MacOS X users (see `here
<http://mac.glueviz.org>`__ for the most recent versions of the app).

When Travis runs (see :doc:`testing`), one of the builds triggers the following
`script <https://github.com/glue-viz/glue/blob/master/.trigger_app_build.sh>`_.
If the Travis build is for a pull request or is not for the main Glue
repository, the script exits early. However, if the test is for the ``master``
branch of the Glue repository, the script runs, and triggers a Travis build for
the `Travis-MacGlue <https://github.com/glue-viz/Travis-MacGlue>`_ repository.

This then sets up a Travis build that includes all the dependencies for Glue,
and then runs `py2app <http://pythonhosted.org/py2app/>`_. Once this has
completed successfully, the app file is uploaded to `Amazon S3
<https://aws.amazon.com/s3/>`_ and becomes available at `mac.glueviz.org
<http://mac.glueviz.org/>`_.

.. TODO provide instructions for how to build app locally
