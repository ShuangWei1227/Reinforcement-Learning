from Wei_Shuang_code_Grid_env import Grid_world
from Wei_Shuang_code_RL_Sarsa import SarsaTable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



list_episode = []
list_rewards = []


def run():
    EPOCHS = 6
    EPISODES = 51
    T = 50   #2len(state)
    for epoch in range(1, EPOCHS):
        
        #initialize state
        state = env.agt_reset_value()


        for episode in range(1, EPISODES):

            env.render()

            #initialize state, action and reward
            state = env.agt_reset_value()

            action = RL.agt_choose(str(state))

            cumulative_gamma = 1

            final_reward = 0

            for timestep in range(1, T):


                #action = RL.agt_choose(str(state))

                #-------deterministic & stochastic version of move function--------
                next_state = env.env_move_det(action)

                reward, done = env.env_reward(next_state)

                #print reward
                final_reward = final_reward + reward
                #print final_reward

                rewards = float(cumulative_gamma * final_reward) / float(EPOCHS)
                

                cumulative_gamma *= 0.99
                #print cumulative_gamma

                next_action = RL.agt_choose(str(next_state))

                #RL learn from this transition
                RL.agt_learn_sarsa(str(state), action, reward, str(next_state), next_action)

                #swap state and action
                state = next_state
                action = next_action

                #print state,action


                #break and record reward
                if done:
                    #print ((epoch-1) * EPISODES + episode)
                    list_episode.append(((epoch-1) * EPISODES + episode))
                    #print list_episode
                    #print rewards
                    list_rewards.append(rewards)
                    #print list_rewards
                    #print '>final_reward =%.2f' % (final_reward)
                    print '>epoch =%d, episode =%d, rewards =%.2f, final_reward =%.2f' % (epoch, episode, rewards, final_reward)
                    plot_sarsa = plt.plot(list_episode, list_rewards, 'pink', lw = 1)
                    break

        #print sum(list_rewards)
        #print list_episode
        #plot_sarsa = plt.plot(list_episode, list_rewards, 'red', lw = 1, label = '0.1-0.1-det-Sarsa')


    # end of game
    print('game over')

    plt.xlim(0,250)
    plt.ylim(-20,100)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend(['1-1-sto-Sarsa'])
    plt.show()
    env.destroy()



if __name__ == "__main__":
    env = Grid_world()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, run)
    env.mainloop()