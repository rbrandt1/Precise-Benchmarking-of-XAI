
from xai_metrics.BaseMetric import BaseMetric

import numpy as np


class Our_CompletenessMetric(BaseMetric):
    def __init__(self, gt_explanations, explanations,operator):
        super().__init__( gt_explanations, explanations,operator)

    def evaluate_single(self, gt,  explanation) -> float:
                   
        mask = np.ones(explanation.shape, dtype='bool')
        
        if self._operator == "!=":
            mask[gt != 0] = False
            explanation_f = abs(abs(explanation)-abs(gt))
            count = np.invert(mask).sum()        
            
        if self._operator == ">":
            mask[gt > 0] = False
            explanation_f = abs(explanation-gt)
            explanation_f[explanation_f > 1] = 1.0
            count = np.invert(mask).sum()   
         
        if self._operator == "<":
            mask[gt < 0] = False
            explanation_f = abs(explanation-gt)
            explanation_f[explanation_f > 1] = 1.0
            count = np.invert(mask).sum() 
           
        sum = np.ma.array(explanation_f, mask = mask).sum()
        if not sum:
            sum = 0.0
        if count == 0:
            solution = 1.0 
        else:
            solution = 1.0 - sum / count
               
        return solution

