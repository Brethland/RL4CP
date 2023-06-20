import numpy as np

from constants import *


alwaysfalse = lambda x: 0


plt.ion()

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



def feedback(elite, score):
    print(f"Best score: {score}")
    display_graph(elite)



def run(agent, score_func, terminal_condition=alwaysfalse, maxreps=1000000):
    for i in range(maxreps):
            
        # generate data
        data = agent.generate(n_sessions)

        # select training data
        data_scores = [score_func(point) for point in data]
        num_elites = round(len(data_scores)*(1.0-percentile/100))
        elites = [data for _, data in sorted(zip(data_scores, data)][:-num_elites] # first sort based on data_scores, then take last num_elites elements

        if terminal_condition(max(data_scores)):
            print(f"Convergence reached! Score: {max(data_scores)}")
            feedback()
            exit()

        # train data

        #!!!
        train_data = torch.from_numpy(np.column_stack((elite_states, elite_actions)))
        train_data = train_data.to(torch.float)
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=32)
        
        agent.train(elites )# !!!!

        # user feedback
        feedback(elites[-1],max(data_scores))

    print(f"Convergence not reached with parameters N={N}, Learningrate={LEARNING_RATE}.")
    return 0

