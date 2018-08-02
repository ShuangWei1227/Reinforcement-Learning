import numpy as np
import pandas as pd


class SarsaTable:
    def __init__(self, actions):
        #actions = [0, 1, 2, 3] means up,down,left,right in grid_evn.Grid_world
        self.actions = actions
        #learning rate
        self.alpha = 0.1
        #reward discount
        self.gamma = 0.99
        #e-greedy
        self.epsilon = 0.1
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def agt_choose(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.rand() < self.epsilon:
            #get random action
            action = np.random.choice(self.actions)
        else:
            #get action from the Q function table
            state_action = self.q_table.ix[state, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        return action



    def agt_learn_sarsa(self, state, action, reward, next_state, next_action):
        self.check_state_exist(next_state)
        q_current = self.q_table.ix[state, action]
        #Bellman Optimality Equation
        #if done != True:
        q_new = reward + self.gamma * self.q_table.ix[next_state, next_action]
        #else:
        #q_new = reward
        #update Q table
        self.q_table.ix[state, action] += self.alpha * (q_new - q_current)
        #return self.q_table



    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index = self.q_table.columns,
                    name = state,))
