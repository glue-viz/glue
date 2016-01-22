"""
Widgets for sending feedback reports
"""
from glue.external.six.moves.urllib.request import Request, urlopen
from glue.external.six.moves.urllib.parse import urlencode
import sys

from glue.external.qt.QtGui import QTextCursor
from glue.qt.qtutil import load_ui

__all__ = ['submit_bug_report']


def _send_feedback(report):
    """
    Send a report to bugs.glueviz.org

    :param report: Report message to send
    :type report: str
    """

    # website expects a post request with a report and specific key
    url = 'http://bugs.glueviz.org'
    values = dict(report=report, key='72z29Q9BzM8sgATeQdu4')

    data = urlencode(values)
    req = Request(url, data)
    urlopen(req)


def _diagnostics():
    """
    Return a some system informaton useful for debugging
    """
    from glue.external.qt import QtCore
    from matplotlib import __version__ as mplversion
    from numpy import __version__ as npversion
    from astropy import __version__ as apversion

    result = []
    result.append('Platform: %s' % sys.platform)
    result.append('Version: %s' % sys.version)
    result.append('Qt Binding: %s' % QtCore.__name__.split('.')[0])
    result.append('Matplotlib version: %s' % mplversion)
    result.append('Numpy version: %s' % npversion)
    result.append('AstroPy version: %s' % apversion)
    return '\n'.join(result)


class FeedbackWidget(object):

    """
    A Dialog to enter and send feedback
    """

    def __init__(self, feedback='', parent=None):
        """
        :param feedback: The default feedback report
        :type feedback: str

        Feedback will be supplemented with diagnostic system information.
        The user can modify or add to any of this
        """
        self._ui = load_ui('feedbackwidget', None)
        feedback = '\n'.join(['-' * 80,
                              feedback,
                              _diagnostics()])
        self._ui.report_area.insertPlainText('\n' + feedback)
        self._ui.report_area.moveCursor(QTextCursor.Start)

    def exec_(self):
        """
        Show and execute the dialog.

        :returns: True if the user clicked "OK"
        """
        self._ui.show()
        self._ui.raise_()
        return self._ui.exec_() == self._ui.Accepted

    @property
    def report(self):
        """
        The contents of the report window
        """
        return self._ui.report_area.document().toPlainText()


def submit_bug_report(report=''):
    """
    Present a user interface for modifying and sending a feedback message

    :param report: A default report message
    :type report: str

    :returns: True if a report was submitted
    """
    widget = FeedbackWidget(report)
    if widget.exec_():
        _send_feedback(widget.report)
        return True
    return False
