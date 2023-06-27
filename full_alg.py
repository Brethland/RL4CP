import numpy as np
import networkx as nx
from time import time
import torch
from torch import nn
import itertools
import matplotlib.pyplot as plt
from math import ceil


#############
# CONSTANTS #
#############


N = 4   #number of vertices in the graph. Only used in the reward function, not directly relevant to the algorithm 
MYN = int(N*(N-1)/2)  #The length of the word we are generating. Here we are generating a graph, so we create a 0-1 word of length (N choose 2)

LEARNING_RATE = 0.0003 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 300 #number of new sessions per iteration
percentile = 93 #top 100-X percentile we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration

FIRST_LAYER_NEURONS = 32 #Number of neurons in the hidden layers.
SECOND_LAYER_NEURONS = 12
THIRD_LAYER_NEURONS = 4

n_actions = 2 #The size of the alphabet. In this colab we will assume this is 2. There are a few things we need to change when the alphabet size is larger,
              #such as one-hot encoding the input, and using categorical_crossentropy as a loss function.
              
observation_space = 2*MYN #The input vector will have size 2*MYN, where the first MYN letters encode our partial word (with zeros on
                          #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                          #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.

len_game = MYN #each game will have this many steps
state_dim = (observation_space,)


#########
# TOOLS #
#########


def speedncount(func):
    def wrap_func(*args, **kwargs):
        wrap_func.calls += 1
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Func {func.__name__!r} exec in {(t2-t1):.4f}s, total of {wrap_func.time}s and {wrap_func.calls} calls')
        wrap_func.time += t2-t1

        return result
    wrap_func.calls = 0
    wrap_func.time = 0
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




###########
# NETWORK #
###########


class Agent:
    def __init__(self, net, optimizer, training_func):
        self.net = net
        self.optimizer = optimizer
        self.training_func = training_func


    @speedncount
    def generate(self, num_sessions):
        @speedncount
        def generate_next_step(actions, i=0):
            if i < len(actions):
                step = np.zeros(len(actions))
                step[i] += 1
                # The net takes a vector of length MYN*2 of the form [actions, step] with step being
                # of the form [0,...,1,...,0] with a 1 on the current index to be generated.
                prob = self.net(torch.from_numpy(np.array([np.concatenate([actions,step])])).to(torch.float))
                prob = prob.detach().cpu().numpy()
                actions[i] = (np.random.rand() > prob) # Sample directly
                return generate_next_step(actions, i=i+1)
            return actions
        
        # Return a vector full with num_sessions numpy arrays that correspond to graphs.
        return [generate_next_step(np.zeros(MYN)) for j in range(num_sessions)]    


    # @speedncount
    def train(self, train_loader, **kwargs):
        self.training_func(self.net, self.optimizer, train_loader, **kwargs)
        



class DenseNet(nn.Module):
    def __init__(self, widths):
        super().__init__()

        num_layers = len(widths)
        layers = [[nn.Linear(widths[i], widths[i+1]), nn.ReLU()] for i in range(num_layers-2)]
        self.layers = [nn.Flatten(1, -1), 
                      *list(itertools.chain(*layers)), 
                      nn.Linear(widths[-2], widths[-1]),
                      nn.Sigmoid()]
                      
        self.net = nn.Sequential(*self.layers)

    
    def forward(self, x):
        prob = self.net(x)
        return prob



# @speedncount
def standard_training(model, optimizer, train_loader,
                  num_epochs=1, pbar_update_interval=200, print_logs=False):
    '''
    Updates the model parameters (in place) using the given optimizer object.
    Returns `None`.
    '''
    criterion = nn.BCELoss()
    pbar = trange(num_epochs) if print_logs else range(num_epochs)

    for i in pbar:
        for k, batch_data in enumerate(train_loader):
            batch_x = batch_data[:, :-1]
            batch_y = batch_data[:, -1]
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y.unsqueeze(1))
            loss.backward() 
            optimizer.step() 

            if print_logs and k % pbar_update_interval == 0:
                acc = (y_pred.round() == batch_y).sum().float()/(len(batch_y))
                pbar.set_postfix(loss=loss.item(), acc=acc.item())


model = DenseNet([2*MYN, FIRST_LAYER_NEURONS, SECOND_LAYER_NEURONS, THIRD_LAYER_NEURONS, 1])
standard_net = Agent(model, torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), standard_training)



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
        print(agent.generate.calls, agent.generate.time/agent.generate.calls)    
        exit()
    
    print(f"Convergence not reached with parameters N={N}, Learningrate={LEARNING_RATE}.")
    return 0


########
# MAIN #
########

# @speedncount
def conjecture2(actions):
    graph = nx.Graph(actions_to_adj(actions))
    if not nx.is_connected(graph):
        return 0
    D = nx.diameter(graph)
    return 10-(Randic_index(actions_to_adj(actions))-D- np.sqrt(2)+ (N+1)/2)

def terminal_condition_conjecture2(score):
    return (score>10)


run(standard_net, conjecture2, terminal_condition=terminal_condition_conjecture2, maxreps=10000)

