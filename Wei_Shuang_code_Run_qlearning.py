from Wei_Shuang_code_Grid_env import Grid_world
from Wei_Shuang_code_RL_Qlearning import QlearningTable

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

            #initialize action
            state = env.agt_reset_value()

            action = RL.agt_choose(str(state))

            cumulative_gamma = 1

            final_reward = 0

            for timestep in range(1, T):

                #action = RL.agt_choose(str(state))

                # RL take action and get next state and reward
                #-------deterministic & stochastic version of move function--------
                next_state = env.env_move_det(action)
                
                reward, done = env.env_reward(next_state)

                #print reward

                final_reward = final_reward + reward
                #print final_reward

                rewards = float(cumulative_gamma * final_reward) / float(EPOCHS)

                #list_episode.append(episode)
                #list_rewards.append(rewards)

                cumulative_gamma *= 0.99

                next_action = RL.agt_choose(str(next_state))    #but without using in Qlearning move function

                # RL learn from this transition
                RL.agt_learn_q(str(state), action, reward, str(next_state))

                #swap state and action
                state = next_state
                action = next_action    #but without using in Qlearning move function

                #break and record reward
                if done:
                    #print ((epoch-1) * EPISODES + episode)
                    list_episode.append(((epoch-1) * EPISODES + episode))
                    #print list_episode
                    #print rewards
                    list_rewards.append(rewards)
                    #print list_rewards
                    print '>epoch =%d, episode =%d, rewards =%.2f, final_reward =%.2f' % (epoch, episode, rewards, final_reward)
                    plot_q = plt.plot(list_episode, list_rewards, 'red', lw = 1)
                    break


    # end of game
    print('game over')

    plt.xlim(0,250)
    plt.ylim(-20,100)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.legend(['0.1-0.1-det-Q'])
    plt.show()
    env.destroy()



if __name__ == "__main__":
    env = Grid_world()
    RL = QlearningTable(actions=list(range(env.n_actions)))

    env.after(100, run)
    env.mainloop()