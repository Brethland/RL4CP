import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import ceil
import torch 

from constants import *
from tools import *


###################
# PROCESS MANAGER #
###################


alwaysfalse = lambda x: 0

plt.ion()

# @speedncount
def display_graph(adjMatG):
    print("Best adjacency matrix in current step:")
    print(adjMatG)

    G = nx.convert_matrix.from_numpy_array(adjMatG)

    plt.clf()
    nx.draw_circular(G)

    plt.axis('equal')
    plt.draw()
    plt.pause(0.001)
    plt.show()

# @speedncount
def feedback(elite, score):
    print(f"Best score: {score}")
    display_graph(actions_to_adj(elite))


# @speedncount
def run(agent, score_func, terminal_condition=alwaysfalse, maxreps=100000):
    old_elites = np.array([]) # These are the Superstates.
    for i in range(maxreps):
            
        # generate data
        data = agent.generate(n_sessions)
        data = np.concatenate((data, old_elites)) if old_elites.size else data

        # select training data
        data_scores = [score_func(point) for point in data]
        num_elites = ceil(len(data_scores)*(1.0-percentile/100))
        
        elites = np.array([data for _, data in sorted(zip(data_scores, data),key=lambda x: x[0])][-num_elites:]) # first sort based on data_scores, then take last num_elites elements

        if terminal_condition(max(data_scores)):
            print(f"Convergence reached! Score: {max(data_scores)}")
            feedback(elites[-1],max(data_scores))
            exit()

        # train data
        train_data = np.zeros((elites.shape[0]*elites.shape[1], elites.shape[1]*2+1))

        def row(i): 
            a = np.zeros(MYN)
            a[i]+=1
            return a

        for i,elite in enumerate(elites):
            for j in range(0, MYN):
                train_data[i*(MYN)+j] = np.concatenate((elite[:j], np.zeros(MYN-j), row(j), np.array([elite[j]])))

        train_data = torch.from_numpy(train_data).to(torch.float)

        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
        agent.train(train_loader)

        # user feedback
        feedback(elites[-1],max(data_scores))

        # super states
        num_old_elites = round(len(data_scores)*(1.0-super_percentile/100.0))
        if num_old_elites:
            old_elites = elites[-num_old_elites:]
        else: old_elites = np.array([]) 
        # print(agent.generate.calls, agent.generate.time/agent.generate.calls)    
        # exit()
    
    print(f"Convergence not reached with parameters N={N}, Learningrate={LEARNING_RATE}.")
    return 0

