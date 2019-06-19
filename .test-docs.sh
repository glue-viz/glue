#!/bin/bash -x

cd doc

make spelling
make clean

make html linkcheck 2> warnings.log

cat warnings.log
grep -v "numpy.dtype size changed" warnings.log \
    | grep -v "numpy.ufunc size changed" \
    | grep -v "return f(\*args, \*\*kwds)" \
    > warnings_filt.log

# make sure stderr was empty, i.e. no warnings
cat warnings_filt.log
test ! -s warnings_filt.log

  # Check for any broken links, ignore 'redirected with Found'
grep -v "redirected with Found" _build/linkcheck/output.txt > _build/linkcheck/output_no_found_redirect.txt

# Make sure file is empty, i.e. no warnings/errors
cat _build/linkcheck/output_no_found_redirect.txt
test ! -s _build/linkcheck/output_no_found_redirect.txt
