import torch
from torch import nn
import numpy as np
import itertools

from constants import *
from tools import speedncount



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
        def generate_next_step(actions, i=0): # Actions has to be a Matrix with the _rows_ being an actions vector
            if i < actions.shape[1]:
                step = np.zeros(actions.shape)
                step[:,i] += 1
                # The net takes a vector of length MYN*2 of the form [actions, step] with step being
                # of the form [0,...,1,...,0] with a 1 on the current index to be generated.
                prob = self.net(torch.from_numpy(np.hstack((actions, step))).to(torch.float))
                prob = prob.detach().cpu().numpy()
                for j in range(actions.shape[0]):
                    actions[j,i] = (np.random.rand() > prob[j]) # Sample directly
                return generate_next_step(actions, i=i+1)
            return actions
        
        # Return a vector full with num_sessions numpy arrays that correspond to graphs.
        return generate_next_step(np.zeros((num_sessions,MYN)))    


    # @speedncount
    def train(self, train_loader, **kwargs):
        self.training_func(self.net, self.optimizer, train_loader, **kwargs)
        


@speedncount
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
