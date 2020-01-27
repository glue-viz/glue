from urllib.request import urlopen

__all__ = ['require_data']

DATA_REPO = "https://raw.githubusercontent.com/glue-viz/glue-example-data/master/"


def require_data(file_path):
    """
    Download the specified file to the current folder, preserving the directory
    structure.

    Note that this should include forward slashes for paths even on Windows.
    """

    # We use urlopen instead of urlretrieve to have control over the timeout

    local_path = file_path.split('/')[-1]

    request = urlopen(DATA_REPO + file_path, timeout=60)
    with open(local_path, 'wb') as f:
        f.write(request.read())

    print("Successfully downloaded data file to {0}".format(local_path))
