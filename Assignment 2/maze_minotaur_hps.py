import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -1
    EATEN_REWARD = 0


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = np.array([0, 0]);
        actions[self.MOVE_LEFT]  = np.array([0,-1]);
        actions[self.MOVE_RIGHT] = np.array([0, 1]);
        actions[self.MOVE_UP]    = np.array([-1,0]);
        actions[self.MOVE_DOWN]  = np.array([1,0]);
        return actions;

    def __states(self):
        states = dict();
        states_vec = dict();

        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = np.array([i,j,k,l]);
                            states_vec[(i,j,k,l)] = s;
                            s += 1;
        return states, states_vec

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.

        list_minotaur_pos = self.__minotaur_positions(state)
        new_minotaur_pos = list_minotaur_pos[np.random.randint(len(list_minotaur_pos))]
        eaten = all(self.states[state][0:2] == self.states[state][2:])
        won = all(self.states[state][0:2] == np.array([6,5]))
        if hitting_maze_walls or eaten or won:
            return state;
        else:
            return self.map[(row, col, new_minotaur_pos[0], new_minotaur_pos[1])];
    def __minotaur_positions(self, state):
        """
            Input: The state as an int
            Returns: A list of possible new minotaur positions from current state 
        """
        minotaur_pos = self.states[state][2:]
        list_pos = []
        width, height = self.maze.shape[0],self.maze.shape[1]
        for a in range(1,5):
        #below is when it is allowed to stand still
        #for a in range(5):
            new_pos = self.actions[a] + minotaur_pos
            if 0<=new_pos[0]<width and 0<=new_pos[1]<height:
                list_pos.append(new_pos)
        return list_pos
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                list_pos = self.__minotaur_positions(s)
                for minotaur_pos in list_pos:
                    next_s = self.__move(s,a);
                    new_pos = np.copy(self.states[next_s])
                    new_pos[2:] = minotaur_pos
                    next_s = self.map[tuple(new_pos)]
                    transition_probabilities[next_s, s, a] = 1/len(list_pos);

                    #if we are eaten in the current state, we can't move
                    if all(self.states[s][0:2] == self.states[s][2:]):
                        transition_probabilities[next_s,s,a] = 0
                    #if we are in B, we can't move
                    if all(self.states[s][0:2] == np.array([6,5])):
                        transition_probabilities[next_s,s,a] = 0

        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));


        for s in range(self.n_states):
            list_pos = self.__minotaur_positions(s)
            for a in range(self.n_actions):
                next_s = self.__move(s,a);
                # Reward for hitting a wall
                if s == next_s and a != self.STAY:

                    rewards[s,a] = self.IMPOSSIBLE_REWARD;
                # Reward for reaching the exit
                elif self.maze[tuple(self.states[next_s][0:2])] == 2:
                    rewards[s,a] = self.GOAL_REWARD;
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s,a] = self.STEP_REWARD;
                # Reward for being in danger of being eaten

                rewards[s,a] += self.EATEN_REWARD*(self.states[next_s][0:2] in np.array(list_pos))
        return rewards;

    def simulate(self, start, policy, goal, T=20):
        
        path = list();
        # Initialize current state, next state and time
        t = 1;
        s = self.map[start];
        # Add the starting position in the maze to the path
        path.append(start);
        # Move to next state given the policy and the current state
        next_s = self.__move(s,policy[s]);
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.states[next_s]);
        # Loop while state is not the goal state
        while not all(self.states[s][0:2] == goal) and t<T:
            # Update state
            s = next_s;
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1;

        agent_end_pos = (self.states[s][0],self.states[s][1])
        mino_end_pos = (self.states[s][2],self.states[s][3])

        if agent_end_pos == (6,5) and agent_end_pos != mino_end_pos:
            return path, True
        
        else:
            return path, False


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 50:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    path_player = [tuple(p)[0:2] for p in path]
    path_mino = [tuple(p)[2:] for p in path]
    for i in range(len(path_player)):
        grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path_player[i])].get_text().set_text('Player')

        grid.get_celld()[(path_mino[i])].set_facecolor(LIGHT_RED)
        grid.get_celld()[(path_mino[i])].get_text().set_text('Mino')        
        if i > 0:
            playerHasMoved = path_player[i] != path_player[i-1]
            playerHasWon = path_player[i] == (6,5)
            playerHasLost = path_player[i] == path_mino[i]
            exchangedPos = path_player[i] == path_mino[i-1] and path_mino[i] == path_player[i-1]
            if playerHasWon and not playerHasLost:
                grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path_player[i])].get_text().set_text('We won!')
                grid.get_celld()[(path_player[i-1])].set_facecolor(col_map[maze[path_player[i-1]]])
                grid.get_celld()[(path_player[i-1])].get_text().set_text('')        
                grid.get_celld()[(path_mino[i-1])].set_facecolor(col_map[maze[path_mino[i-1]]])
                grid.get_celld()[(path_mino[i-1])].get_text().set_text('')
                break  
            elif playerHasLost and playerHasMoved:
                grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path_player[i])].get_text().set_text('Player is out')
                grid.get_celld()[(path_player[i-1])].set_facecolor(col_map[maze[path_player[i-1]]])
                grid.get_celld()[(path_player[i-1])].get_text().set_text('')   
                grid.get_celld()[(path_mino[i-1])].set_facecolor(col_map[maze[path_mino[i-1]]])
                grid.get_celld()[(path_mino[i-1])].get_text().set_text('')
                break
            elif playerHasLost:
                grid.get_celld()[(path_player[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path_player[i])].get_text().set_text('Player is out') 
                grid.get_celld()[(path_mino[i-1])].set_facecolor(col_map[maze[path_mino[i-1]]])
                grid.get_celld()[(path_mino[i-1])].get_text().set_text('') 
                break         
            elif exchangedPos:
                pass
            elif playerHasMoved:
                grid.get_celld()[(path_player[i-1])].set_facecolor(col_map[maze[path_player[i-1]]])
                grid.get_celld()[(path_player[i-1])].get_text().set_text('')        
                grid.get_celld()[(path_mino[i-1])].set_facecolor(col_map[maze[path_mino[i-1]]])
                grid.get_celld()[(path_mino[i-1])].get_text().set_text('')
            else:
                grid.get_celld()[(path_mino[i-1])].set_facecolor(col_map[maze[path_mino[i-1]]])
                grid.get_celld()[(path_mino[i-1])].get_text().set_text('')      
        plt.pause(0.3)
    plt.show()

        
maze = np.array([
    [ 0, 0, 1, 0, 0, 0,  0, 0],
    [ 0, 0, 1, 0, 0, 1,  0, 0],
    [ 0, 0, 1, 0, 0, 1,  1, 1],
    [ 0, 0, 1, 0, 0, 1, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])
mz = Maze(maze)
#V, policy = value_iteration(mz, gamma = 0.8, epsilon = 1e-3)
#np.save('policy.npy', policy)

goal_rewards = [0,1,5, 10]
eaten_rewards = [0,-1,-5, -10]
start = (0,0,6,5)
goal = (6,5)

#for this run, best seems to be goal 1 and eaten -5, produces a prob of getting out = 0.574
#this was done with gamma = 1, n (max_iter) = 50

if True:
    for goal_reward in goal_rewards:
        for eaten_reward in eaten_rewards:
            mz = Maze(maze)
            mz.GOAL_REWARD = goal_reward
            mz.EATEN_REWARD = eaten_reward
            V, policy = value_iteration(mz, gamma = 1, epsilon = 1e-3)
            
            np.save('policies/'+str(goal_reward)+'_'+str(eaten_reward),policy)
            policy=np.load('policies/'+str(goal_reward)+'_'+str(eaten_reward)+'.npy')
            num_wins = 0

            for _ in range(2000):
                path, is_win = mz.simulate(start, policy, goal)
                if is_win:
                    num_wins += 1

            print("Goal reward was "+str(goal_reward))
            print("Eaten reward was "+str(eaten_reward))
            print("Win fraction was "+str(num_wins/2000))
            print()


#SO BEST WERE: goal 1 eaten -1 w prob 0.571




if False:
    for T in range(16,22):
        #best policy
        policy=np.load('policies/'+str(1)+'_'+str(-5)+'.npy')
        num_wins = 0
        for _ in range(2000):
            path, is_win = mz.simulate(start, policy, goal,T=T)
            if is_win:
                num_wins+=1

        np.save("win_ratios_T/"+str(T),np.array(num_wins/2000))
        print("T = "+str(T))
        print("Win ratio was "+str(num_wins/2000))
        print()


if False:
    #train with minotaur standing still
    mz = Maze(maze)
    mz.GOAL_REWARD = 1
    mz.EATEN_REWARD = -5
    V, policy = value_iteration(mz, gamma = 1, epsilon = 1e-3)

    np.save('policies/minotaur_still'+str(1)+'_'+str(-5),policy)
    policy=np.load('policies/minotaur_still'+str(1)+'_'+str(-5)+'.npy')

    for T in range(16,22):
        #best policy
        policy=np.load('policies/'+str(1)+'_'+str(-5)+'.npy')
        num_wins = 0
        for _ in range(2000):
            path, is_win = mz.simulate(start, policy, goal,T=T)
            if is_win:
                num_wins+=1

        np.save("win_ratios_T/minotaur_still"+str(T),np.array(num_wins/2000))
        print("T = "+str(T))
        print("Win ratio was "+str(num_wins/2000))
        print()

if False:    #TODO This plot hasn't been created
    Ts = [0 for _ in range(14)]
    for T in range(15,22):
        Ts.append(T)

    probs = [0 for _ in range(14)]
    for T in range(15,22):
        prob = np.load("win_ratios_T/"+str(T)+".npy")
        probs.append(probs)

    plot = plt.plot(Ts,probs)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xabel('T',fontsize=12)
    plt.ylabel('Win probability',fontsize=12)
    plt.savefig(plot)


if False:    #TODO This plot hasn't been created
    Ts = [0 for _ in range(14)]
    for T in range(15,22):
        Ts.append(T)

    probs = [0 for _ in range(14)]
    for T in range(15,22):
        prob = np.load("win_ratios_T/minotaur_still"+str(T)+".npy")
        probs.append(probs)

    plot = plt.plot(Ts,probs)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xabel('T',fontsize=12)
    plt.ylabel('Win probability',fontsize=12)
    plt.savefig(plot)


