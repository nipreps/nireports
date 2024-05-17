class NiReportsException(Exception):
    pass


class ReportletException(NiReportsException):
    pass


class RequiredReportletException(ReportletException):
    def __init__(self, config):
        message = (f"No content found while generated reportlet listed as required with the"
        f"following config: {config}")
        self.args = (message)
