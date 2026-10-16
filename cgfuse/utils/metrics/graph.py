from .metric import Metric

class GraphMatch(Metric):
    def __call__(self, ref, gold):
        raise NotImplementedError("GraphMatch is not implemented")
