

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