# Infrastructure for helpful warnings/dialogs that can be hidden if needed
# (and therefore connect to the settings).

from qtpy.QtWidgets import QMessageBox, QCheckBox

from glue.config import settings
from glue._settings_helpers import save_settings

__all__ = ['info', 'warn']


def info(title, text, setting=None, default=None):
    return dialog(title, text, QMessageBox.Information, setting=setting, default=default)


def warn(title, text, setting=None, default=None):
    return dialog(title, text, QMessageBox.Warning, setting=setting, default=default)


def dialog(title, text, icon, setting=None, default=None):

    if not getattr(settings, setting.upper()):
        return True

    check = QCheckBox()
    check.setText('Don\'t show this message again (can be reset via the preferences)')

    info = QMessageBox()
    info.setIcon(icon)
    info.setText(title)
    info.setInformativeText(text)
    info.setCheckBox(check)
    info.setStandardButtons(info.Cancel | info.Ok)
    if default == 'Cancel':
        info.setDefaultButton(info.Cancel)

    result = info.exec_()

    if result == info.Cancel:
        return False

    if check.isChecked():
        setattr(settings, setting.upper(), False)
        save_settings()

    return True


if __name__ == "__main__":
    from glue.utils.qt import get_qapp
    app = get_qapp()
    info('What happens next?', 'These are instructions on what happens next', setting='show_info_profile_open')
