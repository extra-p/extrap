class Model:
    
    def __init__(self, hypothesis, callpath_id, metric_id):
        self.hypothesis = hypothesis
        self.callpath_id = callpath_id
        self.metric_id = metric_id
        
    def get_hypothesis(self):
        return self.hypothesis
    
    def get_callpath_id(self):
        return self.callpath_id
    
    def get_metric_id(self):
        return self.metric_id