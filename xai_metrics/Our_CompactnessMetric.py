
from xai_metrics.BaseMetric import BaseMetric

import numpy as np


class Our_CompactnessMetric(BaseMetric):
    def __init__(self,  gt_explanations, explanations,operator):
        super().__init__(gt_explanations, explanations,operator)


    def evaluate_single(self, gt, explanation) -> float:
                            
        maskTP = np.ones(explanation.shape, dtype='bool')
        maskFP = np.ones(explanation.shape, dtype='bool')
        
        if self._operator == "!=":

            maskTP[np.logical_and(gt != 0,explanation != 0)] = False
            maskFP[np.logical_and(gt == 0,explanation != 0)] = False
            
            error_exp = abs(abs(explanation) - abs(gt))

        if self._operator == ">":

            maskTP[np.logical_and(gt > 0,explanation > 0)] = False
            maskFP[np.logical_and(gt <= 0,explanation > 0)] = False

            error_exp = abs(explanation - gt) 
            error_exp[error_exp > 1] = 1.0

        if self._operator == "<":

            maskTP[np.logical_and(gt < 0,explanation < 0)] = False
            maskFP[np.logical_and(gt >= 0,explanation < 0)] = False
            
            error_exp = abs(explanation - gt)
            error_exp[error_exp > 1] = 1.0

        top = np.ma.array(np.ones(error_exp.shape) - error_exp, mask = maskTP).sum()
        if not top:
            top = 0.0
        
        fpsum = np.ma.array(error_exp, mask = maskFP).sum()
        if not fpsum:
            fpsum = 0.0
        
        bottom = top + fpsum
        
        if top == 0 and bottom == 0:
            solution = 1.0
        elif top == 0 and bottom != 0:
            solution = 0.0
        else:
            solution = top / bottom

        return solution