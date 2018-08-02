import numpy as np
import time
import sys
import Tkinter as tk



UNIT = 100
grid_height = 5
grid_width = 5


class Grid_world(tk.Tk, object):
    def __init__(self):
        super(Grid_world, self).__init__()
        self.action_options = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_options)
        self.title('grid world')
        self.geometry('{0}x{1}'.format(grid_height * UNIT, grid_height * UNIT))
        self.build_grid()

    def build_grid(self):
        self.canvas = tk.Canvas(self, bg='cyan',
                           height = grid_height * UNIT,
                           width = grid_width * UNIT)

        #create grids
        for c in range(0, grid_width * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, grid_height * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, grid_height * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, grid_height * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        #create origin
        origin = np.array([20, 20])

        #monster
        monster1_center = origin + np.array([UNIT, UNIT * 2])
        self.monster1 = self.canvas.create_rectangle(
            monster1_center[0] - 10, monster1_center[1] - 10,
            monster1_center[0] + 70, monster1_center[1] + 70,
            fill='black')
        monster2_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.monster2 = self.canvas.create_rectangle(
            monster2_center[0] - 10, monster2_center[1] - 10,
            monster2_center[0] + 70, monster2_center[1] + 70,
            fill='black')


        #strawberry
        strawberry1_center = origin + np.array([UNIT * 4, UNIT * 3])
        self.strawberry1 = self.canvas.create_rectangle(
            strawberry1_center[0] - 10, strawberry1_center[1] - 10,
            strawberry1_center[0] + 70, strawberry1_center[1] + 70,
            fill='pink')
        strawberry2_center = origin + np.array([UNIT * 1, UNIT * 3])
        self.strawberry2 = self.canvas.create_rectangle(
            strawberry2_center[0] - 10, strawberry2_center[1] - 10,
            strawberry2_center[0] + 70, strawberry2_center[1] + 70,
            fill='pink')

        #create home
        home_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.home = self.canvas.create_oval(
            home_center[0] - 10, home_center[1] - 10,
            home_center[0] + 70, home_center[1] + 70,
            fill='yellow')

        #create agent
        self.agent = self.canvas.create_rectangle(
            origin[0] - 10, origin[1] - 10,
            origin[0] + 70, origin[1] + 70,
            fill='red')

        # pack all
        self.canvas.pack()

    def agt_reset_value(self):
        self.update()
        time.sleep(0.05)
        self.canvas.delete(self.agent)
        origin = np.array([20, 20])
        self.agent = self.canvas.create_rectangle(
            origin[0] - 10, origin[1] - 10,
            origin[0] + 70, origin[1] + 70,
            fill='red')
        return self.canvas.coords(self.agent)


    #deterministic version
    def env_move_det(self, action):
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        self.render()
        #up
        if action == 0:
            if state[1] > UNIT:
                base_action[1] -= UNIT
        #down
        elif action == 1:
            if state[1] < (grid_height - 1) * UNIT:
                base_action[1] += UNIT
        #left
        elif action == 2:
            if state[0] > UNIT:
                base_action[0] -= UNIT
        #right
        elif action == 3:
            if state[0] < (grid_width - 1) * UNIT:
                base_action[0] += UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])
        
        next_state = self.canvas.coords(self.agent)

        return next_state


    #stochastic version
    def env_move_sto(self, action):
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        self.render()
        trans_pro = np.random.rand()
        #up
        if action == 0:
            if state[1] > UNIT:
                if trans_pro < 0.8:   #up
                    base_action[1] -= UNIT
                elif trans_pro > 0.9: #left
                    base_action[0] -= UNIT
                else:                 #right
                    base_action[0] += UNIT

        #down
        elif action == 1:
            if state[1] < (grid_height - 1) * UNIT:
                if trans_pro < 0.8:   #down
                    base_action[1] += UNIT
                elif trans_pro > 0.9: #left
                    base_action[0] -= UNIT
                else:                 #right
                    base_action[0] += UNIT

        #left
        elif action == 2:
            if state[0] > UNIT:
                if trans_pro < 0.8:   #left
                    base_action[0] -= UNIT
                elif trans_pro > 0.9: #up
                    base_action[1] -= UNIT
                else:                 #down
                    base_action[1] += UNIT
        #right
        elif action == 3:
            if state[0] < (grid_width - 1) * UNIT:
                base_action[0] += UNIT
                if trans_pro < 0.8:   #right
                    base_action[0] += UNIT
                elif trans_pro > 0.9: #up
                    base_action[1] -= UNIT
                else:                 #down
                    base_action[0] += UNIT


        self.canvas.move(self.agent, base_action[0], base_action[1])
        
        next_state = self.canvas.coords(self.agent)

        return next_state


    #reward function
    def env_reward(self, next_state):

        if next_state == self.canvas.coords(self.home):
            reward = 100
            done = True
        elif next_state in [self.canvas.coords(self.monster1), self.canvas.coords(self.monster2)]:
            reward = -10
            done = False
        elif next_state in [self.canvas.coords(self.strawberry1),self.canvas.coords(self.strawberry2)]:
            reward = 20
            done = False
        else:
            reward = 0
            done = False

        return reward, done


    def render(self):
        time.sleep(0.05)
        self.update()


if __name__ == '__main__':
    env = Grid_world()
    #env.after(100, update)
    env.mainloop()


