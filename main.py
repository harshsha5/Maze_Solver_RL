import aid_gym
import numpy as np
import ipdb

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
goal_state = env.s_goal

print(env.MAP_SIZE)
V_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE))

# valid_states = np.zeros((1,2))
valid_state_list = []

##Modifying the Value function table for obstacles and goal states
print("Modifying V_table")
for i in range(env.MAP_SIZE):
  for j in range(env.MAP_SIZE):
    pos = np.array([i,j]) 
    if(not env._isValidPosition(pos)):
        V_table[i][j] = -float('Inf')
    else:
        print(i,j)
        # valid_states = np.vstack((valid_states,pos))
        valid_state_list.append(pos)

ipdb.set_trace()

V_table[goal_state[0]][goal_state[1]] = 1000
print("Modified V_table")

total_reward = 0

threshold_value = 0.5
max_iteration_count = 1
gamma = 0.95
has_converged = True

dirX = [-1,-1,-1,0,0,1,1,1]
dirY = [-1,0,1,-1,1,-1,0,1]

for present_iteration_count in range(max_iteration_count):
  has_converged = True  
  for i in range(env.MAP_SIZE):
    for j in range(env.MAP_SIZE):
          env.s = np.array([i,j])
          print(env.s)
          if(not env._isValidPosition(env.s)):
            # print("State is invalid")
            # ipdb.set_trace()
            continue
          else:
            max_Q = -float('Inf')
            for elt in range(len(dirX)):
                action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
                next_state, reward, done, info = env.step(action)
                env.s = env.s = np.array([i,j])  #Since step function changes the present state. We don't want that here
                success_prob =  env._getSuccessProbability(env._getHeightDifference(env.s, next_state[:2]))
                Q = reward + gamma*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[i][j])
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
    for elt in range(len(dirX)):
        action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
        s_ = np.clip(env.s + np.array(action), 0, env.MAP_SIZE-1)
        # next_state, reward, done, info = env.step(action)
        # env.s = curr_state
        if(V_table[s_[0]][s_[1]]>best_V_val):
            best_action = action

    print("Next state is: ",next_state,"\t","Present_state is: ",env.s,"\t Action is: ",best_action)
    if(not best_V_val>-float('Inf')):
        print("No Route to goal exists")
    else:
       next_state, reward, done, info = env.step(action) 

    if(done and next_state[2:] == goal_state):
        print("Path to goal found. Goal reached successfully")
    else:
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