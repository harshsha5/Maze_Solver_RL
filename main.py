import aid_gym
import numpy as np
import ipdb

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
goal_state = env.s_goal

print(env.MAP_SIZE)
print("start_state is: ",start_state,"\t","Goal State: ",goal_state)
V_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE))
obstacle_table = np.full((env.MAP_SIZE,env.MAP_SIZE), False, dtype=bool)

# # valid_states = np.zeros((1,2))
# valid_state_list = []

##Modifying the Value function table for obstacles and goal states
"""Uncomment this section before submitting"""

# print("Modifying V_table")
# for i in range(env.MAP_SIZE):
#   for j in range(env.MAP_SIZE):
#     pos = np.array([i,j]) 
#     if(not env._isValidPosition(pos)):
#         V_table[i][j] = -float('Inf')
#         obstacle_table[i][j] = True

# print("Modified V_table")
# np.save("initially_modified_v_table", V_table)
# np.save("obstacle_table", obstacle_table)
# print("Saved v_table and obstacle _table")
obstacle_table = np.load('obstacle_table.npy')
V_table = np.load('initially_modified_v_table.npy')
ipdb.set_trace()

V_table[goal_state[0]][goal_state[1]] = 1000

total_reward = 0

threshold_value = 0.5
max_iteration_count = 5
gamma = 0.9
has_converged = True

dirX = [-1,-1,-1,0,0,1,1,1]
dirY = [-1,0,1,-1,1,-1,0,1]

for present_iteration_count in range(max_iteration_count):
  has_converged = True  
  for i in range(env.MAP_SIZE):
    for j in range(env.MAP_SIZE):
          env.s = np.array([i,j])
          if(obstacle_table[i][j]):
            continue
          else:
            if(i%100==0):
                print(i,j)

            max_Q = -float('Inf')
            for elt in range(len(dirX)):
                action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
                next_state, reward, done, info = env.step(action)   #Change this. Don't take step. Instead actually figure out next state
                env.s = np.array([i,j])  #Since step function changes the present state. We don't want that here
                # env.s = valid_state
                success_prob =  env._getSuccessProbability(env._getHeightDifference(env.s, next_state[:2]))
                Q = reward + gamma*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[i][j])
                # Q = reward + gamma*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[valid_state[0]][valid_state[1]])
                if(Q>max_Q):
                    max_Q = Q

            old_V = V_table[i][j]
            V_table[i][j] = max_Q 

            if(has_converged and abs(old_V-max_Q)>threshold_value):
                has_converged = False

  if(has_converged):
    print("Value functions have converged")
    break

print("Iterations completed")

env.s = start_state
done = False
while not done:
    best_V_val = -float('Inf')
    curr_state = env.s
    for elt in range(len(dirX)):
        action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
        s_ = np.clip(env.s + np.array(action), 0, env.MAP_SIZE-1)
        # next_state, reward, done, info = env.step(action)
        # env.s = curr_state
        if(V_table[s_[0]][s_[1]]>best_V_val):
            best_action = action
            best_V_val = V_table[s_[0]][s_[1]]

    if(not best_V_val>-float('Inf')):
        print("No Route to goal exists")
    else:
        next_state, reward, done, info = env.step(best_action) 
        print("Next state is: ",next_state[:2],"\t","Present_state is: ",curr_state,"\t Action is: ",best_action)

    if(done and next_state[2:] == goal_state):
        print("Path to goal found. Goal reached successfully")
    elif(done):
        print("Robot fell into water")
           
# for i in range(1000):
#     a = env.actionSpace.sample()
#     state, reward, done, info = env.step(a)
#     total_reward += reward
#     # print(i, state, reward)
#     if done:
#         break

# print(state, total_reward)

""" Test with smaller map"""
""" Form list of valid cells and iterate over them"""


"Assumptions: The robot can't stand still at a place"

"Doubts: How does the goal state get a high value function. Should we initialize it to 1000?"
"Can we somehow improve the process for getting rid of the water/untraversable states without having to call that function again. Maybe maintain a set?"