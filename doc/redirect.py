# This Sphinx plugin comes from https://github.com/openstack/nova-specs and was
# originally licensed under a Creative Commons Attribution 3.0 Unported License.
# The full text for this license can be found here:
#
# http://creativecommons.org/licenses/by/3.0/legalcode


# A simple sphinx plugin which creates HTML redirections from old names
# to new names. It does this by looking for files named "redirect" in
# the documentation source and using the contents to create simple HTML
# redirection pages for changed filenames.

import os.path

from sphinx.application import ENV_PICKLE_FILENAME
from sphinx.util.console import bold


def setup(app):
    from sphinx.application import Sphinx
    if not isinstance(app, Sphinx):
        return
    app.connect('build-finished', emit_redirects)


def process_redirect_file(app, path, ent):
    parent_path = path.replace(app.builder.srcdir, app.builder.outdir)
    with open(os.path.join(path, ent)) as redirects:
        for line in redirects.readlines():
            from_path, to_path = line.rstrip().split(' ')
            from_path = from_path.replace('.rst', '.html')
            to_path = to_path.replace('.rst', '.html')

            redirected_filename = os.path.join(parent_path, from_path)
            redirected_directory = os.path.dirname(redirected_filename)
            if not os.path.exists(redirected_directory):
                os.makedirs(redirected_directory)
            with open(redirected_filename, 'w') as f:
                f.write('<html><head><meta http-equiv="refresh" content="0; '
                        'url=%s" /></head></html>'
                        % to_path)


def emit_redirects(app, exc):
    app.builder.info(bold('scanning %s for redirects...') % app.builder.srcdir)

    def process_directory(path):
        for ent in os.listdir(path):
            p = os.path.join(path, ent)
            if os.path.isdir(p):
                process_directory(p)
            elif ent == 'redirects':
                app.builder.info('   found redirects at %s' % p)
                process_redirect_file(app, path, ent)

    process_directory(app.builder.srcdir)
    app.builder.info('...done')
