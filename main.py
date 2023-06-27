import numpy as np
import networkx as nx


from constants import *
from tools import *
from process_manager import run
from network import standard_net



# def conjecture3(actions):
#     mat = actions_to_adj(actions)
#     return (Randic_index(mat) + girth(nx.Graph(mat)) - 3.5)*np.sqrt(N-1)/(n-3+np.sqrt(2))
    

@speedncount
def conjecture2(actions):
    graph = nx.Graph(actions_to_adj(actions))
    if not nx.is_connected(graph):
        return 0
    D = nx.diameter(graph)
    return 10-(Randic_index(actions_to_adj(actions))-D- np.sqrt(2)+ (N+1)/2)

def terminal_condition_conjecture2(score):
    return (score>10)


run(standard_net, conjecture2, terminal_condition=terminal_condition_conjecture2, maxreps=10000)

