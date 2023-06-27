import numpy as np
import networkx as nx
from time import time

from constants import *

def speedncount(func):
    def wrap_func(*args, **kwargs):
        wrap_func.calls += 1
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Func {func.__name__!r} exec in {(t2-t1):.4f}s')
        return result
    wrap_func.calls = 0
    return wrap_func

def actions_to_adj(actions):
    adjMat = np.zeros((N,N),dtype=np.int8)
    count = 0
    for i in range(N):
        for j in range(i+1,N):
            if actions[count] == 1:
                adjMat[i][j], adjMat[j][i] = 1, 1
            count += 1
    return adjMat


def Randic_index(adjMat):
    R=0 # Randic index
    for i in range(adjMat.shape[0]):
        for j in range(i+1,adjMat.shape[1]):
            if adjMat[i,j]==1:
                R += 1/np.sqrt(sum(adjMat[i])*sum(adjMat[j]))
    return R
