import numpy as np
import qubovert
from qubovert.problems import Problem

def problem_to_qubo_matrix(problem:Problem) -> np.ndarray:
    """
    Convert a qubovert problem to a numpy matrix
    """
    prob_dic = dict(problem.to_qubo())
    n = max(max(k) for k in prob_dic.keys() if k) + 1
    Q = np.zeros((n, n))
    for ij in prob_dic.keys():
        if len(ij) == 2:
            i, j = ij
            Q[i][j] = prob_dic[ij]
            Q[j][i] = prob_dic[ij]
        elif len(ij) == 1:
            i = ij[0]
            Q[i][i] = prob_dic[ij]
        
    return Q
