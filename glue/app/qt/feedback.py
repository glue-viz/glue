"""
Widgets for sending feedback reports
"""
import os

from qtpy import QtGui, QtWidgets
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from glue.utils.qt import load_ui
from glue._deps import get_status_as_odict


__all__ = ['submit_bug_report', 'submit_feedback']


def diagnostics():
    """
    Return a some system informaton useful for debugging
    """
    versions = ""
    for package, version in get_status_as_odict().items():
        versions += "* {0}: {1}\n".format(package, version)
    return versions.strip()


class BaseReportWidget(QtWidgets.QDialog):

    def accept(self):
        """
        Send a report to bugs.glueviz.org
        """

        # website expects a post request with a report and specific key
        url = 'http://bugs.glueviz.org'

        values = dict(report=self.content, key='72z29Q9BzM8sgATeQdu4')

        data = urlencode(values)

        req = Request(url, data.encode('utf-8'))
        urlopen(req)

        self.close()

    @property
    def comments(self):
        return self.ui.area_comments.document().toPlainText() or "No comments"

    @property
    def email(self):
        return self.ui.value_email.text() or "Not provided"


FEEDBACK_TEMPLATE = """
Email address: {email}

Comments
--------

{comments}

System information
------------------

{report}
"""


class FeedbackWidget(BaseReportWidget):
    """
    A Dialog to enter and send feedback
    """

    def __init__(self, parent=None):

        super(FeedbackWidget, self).__init__(parent=parent)

        self.ui = load_ui('report_feedback.ui', self,
                          directory=os.path.dirname(__file__))

        self.ui.area_comments.moveCursor(QtGui.QTextCursor.Start)

    @property
    def report(self):
        if self.ui.checkbox_system_info.isChecked():
            return diagnostics()
        else:
            return "No version information provided"

    @property
    def content(self):
        """
        The contents of the feedback window
        """
        return FEEDBACK_TEMPLATE.format(email=self.email,
                                        comments=self.comments,
                                        report=self.report)


REPORT_TEMPLATE = """
Email address: {email}

Comments
--------

{comments}

Report
------

{report}
"""


class CrashReportWidget(BaseReportWidget):
    """
    A dialog to report crashes/errors
    """

    def __init__(self, crash_report='', parent=None):
        """
        :param feedback: The default feedback report
        :type feedback: str

        Feedback will be supplemented with diagnostic system information.
        The user can modify or add to any of this
        """

        super(CrashReportWidget, self).__init__(parent=parent)

        self.ui = load_ui('report_crash.ui', self,
                          directory=os.path.dirname(__file__))

        self.ui.area_report.insertPlainText(diagnostics() + "\n\n" + crash_report)
        self.ui.area_comments.moveCursor(QtGui.QTextCursor.Start)

    @property
    def report(self):
        return self.ui.area_report.document().toPlainText() or "No report"

    @property
    def content(self):
        """
        The contents of the feedback window
        """
        return REPORT_TEMPLATE.format(email=self.email,
                                      comments=self.comments,
                                      report=self.report)


def submit_bug_report(report=''):
    """
    Present a user interface for sending a crash report

    Parameters
    ----------
    report : str
        The crash report/trackback
    """
    widget = CrashReportWidget(crash_report=report)
    widget.exec_()


def submit_feedback():
    """
    Present a user interface for modifying and sending a feedback message
    """
    widget = FeedbackWidget()
    widget.exec_()


if __name__ == "__main__":

    from glue.utils.qt import get_qapp
    app = get_qapp()
    submit_bug_report(report="Crash log here")
    submit_feedback()
