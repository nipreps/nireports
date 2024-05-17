class NiReportsException(Exception):
    pass

class ReportletException(NiReportsException):
    pass

class RequiredReportletException(ReportletException):
    def __init__(self, config):
        self.args = (f'No content found while generated reportlet listed as required with the following config: {config}',)
