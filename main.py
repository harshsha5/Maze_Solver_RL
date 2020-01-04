import aid_gym
import numpy as np
import ipdb

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
goal_state = env.s_goal

print(env.MAP_SIZE)
V_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE))

total_reward = 0

threshold_value = 0.5
max_iteration_count = 100
gamma = 0.95

dirX = [-1,-1,-1,0,0,1,1,1]
dirY = [-1,0,1,-1,1,-1,0,1]

for present_iteration_count in range(max_iteration_count):
  for i in range(env.MAP_SIZE):
      for j in range(env.MAP_SIZE):
          env.s = np.array([i,j])   
          print(env.s)
          if(not env._isValidPosition(env.s)):
            print("State is invalid")
            continue
          else:
            max_Q = -float('Inf')
            for elt in range(len(dirX)):
                ipdb.set_trace()
                action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
                next_state, reward, done, info = env.step(action)
                success_prob =  env._getSuccessProbability(env._getHeightDifference(env.s, next_state[:2]))
                Q = reward + gamma*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[i][j])
                if(Q>max_Q):
                    max_Q = Q
            V_table[i][j] = max_Q 

env.s = start_state
while not done:
    for elt in range(len(dirX)):
            action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
            next_state, reward, done, info = env.step(action)


            
# for i in range(1000):
#     a = env.actionSpace.sample()
#     state, reward, done, info = env.step(a)
#     total_reward += reward
#     # print(i, state, reward)
#     if done:
#         break

print(state, total_reward)


"Assumptions: The robot can't stand still at a place"

"Doubts: How does the goal state get a high value function. Should we initialize it to 1000?"
"Can we somehow improve the process for getting rid of the water/untraversable states without having to call that function again. Maybe maintain a set?"