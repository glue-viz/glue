Releasing a new package version
===============================

A new release of packages in the `glue-viz <https://glueviz.org/>`_ ecosystem is
now almost fully automated.
For maintainers it should be nice and simple to do, especially if all merged PRs
have informative titles and are correctly labelled.

Here is the process to follow to create a new release:

#. Go through all the PRs since the last release, make sure they have
   descriptive titles (as these will become the auto-generated changelog entries)
   and are labelled correctly - preferably identifying them as one of
   `bug`, `enhancement` or `documentation`.

#. Go to the GitHub releases interface and draft a new release; new tags should
   include the trailing patch version ``.0`` (e.g. ``1.6.0``, not ``1.6``) on
   `major.minor` releases (early releases of most packages did not).

#. Use the GitHub autochange log generator; this should use the configuration in
   `.github/release.yml <https://github.com/glue-viz/glue/.github/release.yml>`_
   to make headings based on labels.

#. Edit the draft release notes as required, particularly to call out major
   changes at the top.

#. Publish the release.
