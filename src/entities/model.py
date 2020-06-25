from util.deprecation import deprecated


class Model:

    def __init__(self, hypothesis, callpath=None, metric=None):
        self.hypothesis = hypothesis
        self.callpath = callpath
        self.metric = metric

    @deprecated("Use property directly.")
    def get_hypothesis(self):
        return self.hypothesis

    @deprecated("Use property directly.")
    def get_callpath_id(self):
        return self.callpath.id

    @deprecated("Use property directly.")
    def get_metric_id(self):
        return self.metric.id
