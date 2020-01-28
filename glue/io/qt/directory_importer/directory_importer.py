from glue.config import importer
from glue.dialogs.data_wizard.qt import data_wizard


@importer("Import from directory")
def directory_importer():
    return data_wizard(mode='directories')
