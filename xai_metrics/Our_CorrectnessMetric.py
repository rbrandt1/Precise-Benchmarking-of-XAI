
from xai_metrics.BaseMetric import BaseMetric
from xai_metrics.Our_CompletenessMetric import Our_CompletenessMetric
from xai_metrics.Our_CompactnessMetric import Our_CompactnessMetric


class Our_CorrectnessMetric(BaseMetric):
    def __init__(self, completeness: Our_CompletenessMetric, compactness: Our_CompactnessMetric):
        super().__init__(None, None, None)
        self.__completeness = completeness
        self.__compactness = compactness

    def get_score(self) -> float:
        return (self.__compactness.get_score() + self.__completeness.get_score()) / 2.0

