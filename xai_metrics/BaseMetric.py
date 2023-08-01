
class BaseMetric: 
    def __init__(self, gt_explanations, explanations, operator):

        self._gt_explanations = gt_explanations
        self._explanations = explanations
        self._operator = operator
        
        self._scores = None

    def evaluate_single(self, gt, explanation) -> float:
        pass

 
    def get_score(self) -> float:

        self._scores = []
        for gt, explanation in zip(self._gt_explanations, self._explanations):
            score = self.evaluate_single(gt, explanation)
            self._scores.append(score)
        
        return sum(self._scores) / len(self._scores)

