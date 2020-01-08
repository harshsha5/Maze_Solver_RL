import numpy as np
import aid_gym
import ipdb

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
goal_state = env.s_goal

print(env.MAP_SIZE)
print("start_state is: ",start_state,"\t","Goal State: ",goal_state)
action_len = 8 #8-connected motion model given
ipdb.set_trace()
Q_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE,action_len))
obstacle_table = np.full((env.MAP_SIZE,env.MAP_SIZE), False, dtype=bool)

##Modifying the Value function table for obstacles and goal states
print("Modifying V_table")
for i in range(env.MAP_SIZE):
  for j in range(env.MAP_SIZE):
    pos = np.array([i,j]) 
    if(not env._isValidPosition(pos)):
        Q_table[i,j,:] = -float('Inf')
        obstacle_table[i][j] = True

ipdb.set_trace()

def q_learning(env,learning_rate,epsilon,discount_factor,max_)

