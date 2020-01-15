import aid_gym
import numpy as np
import ipdb
import cv2
import time

def display_map(start_state,goal_state):
    start = start_state/5
    start = start.astype(int)
    goal = goal_state/5
    goal = goal.astype(int)
    image_location = "assets/island.png"
    img = cv2.imread(image_location)
    cv2.circle(img,(start[1], start[0]), 1, (255,0,0), -1)
    cv2.circle(img,(goal[1], goal[0]), 1, (0,255,0), -1)
    cv2.imshow("Final_Map", img)
    cv2.waitKey(0)

def update_V_table(V_table,rollout,GAMMA,ERROR_THRESHOLD):
    has_converged = True
    while(len(rollout)!=0):
        curr_state,next_state,reward,success_prob,best_action = rollout.pop()
        old_Q = V_table[curr_state[0]][curr_state[1]]
        V_table[curr_state[0]][curr_state[1]] = reward + GAMMA*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[curr_state[0]][curr_state[1]])
        if(has_converged and abs(V_table[curr_state[0]][curr_state[1]]-old_Q)>ERROR_THRESHOLD):
            has_converged = False
    # print("Updated value function in reverse order!")
    return has_converged

def is_arr_in_list(array, list_of_arrays):
    for a in list_of_arrays:
        if np.array_equal(array, a):
            return True
    return False

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
goal_state = env.s_goal

# display_map(start_state,goal_state)

print(env.MAP_SIZE)
print("Start_state is: ",start_state,"\t","Goal State: ",goal_state)
# ipdb.set_trace()
# V_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE))
# obstacle_table = np.full((env.MAP_SIZE,env.MAP_SIZE), False, dtype=bool)

# # valid_states = np.zeros((1,2))
# valid_state_list = []

##Modifying the Value function table for obstacles and goal states
"""Uncomment this section to recompute V_table and obstacle_table initialization"""

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
# obstacle_table = np.load('obstacle_table.npy')
V_table = np.load('initially_modified_v_table.npy')

V_table[goal_state[0]][goal_state[1]] = 100000

MAXIMUM_EPISODE_LENGTH  = 7*int(np.linalg.norm(start_state-goal_state))
print("MAXIMUM_EPISODE_LENGTH: ",MAXIMUM_EPISODE_LENGTH)
MAX_EPISODES = 25000
HAS_CONVERGED = False
GAMMA = 1
ERROR_THRESHOLD = 0.80

dirX = [-1,-1,-1,0,0,1,1,1]
dirY = [-1,0,1,-1,1,-1,0,1]

curr_state = start_state
done = False

if(np.array_equal(start_state, goal_state)):
    print("Goal already reached")
    exit()

t0 = time.time()

num_episodes=0
last_feasible_rollout = []
REACHED_GOAL_ATLEAST_ONCE = False
potential_reward = -float('Inf')
while(num_episodes<MAX_EPISODES):
    num_episodes+=1
    print("Episode: ",num_episodes)
    episode_length = 0
    curr_state = start_state
    done = False
    HAS_CONVERGED = True
    rollout = []
    potential_feasible_rollout = []
    visited = [start_state]
    tot_reward = 0
    while(not done and episode_length<MAXIMUM_EPISODE_LENGTH):
        max_Q = -float('Inf')
        for elt in range(len(dirX)):
            action = [dirX[elt],dirY[elt]]                    
            next_state = np.clip(curr_state + np.array(action), 0, env.MAP_SIZE-1)
            reward = -np.sqrt(np.square(action).sum())      #See if I should handle reward for goal state or is value function enough for that
            success_prob =  env._getSuccessProbability(env._getHeightDifference(curr_state, next_state[:2]))
            # if(np.array_equal(next_state[:2], goal_state)):
            #     reward+=success_prob*1000
            Q = reward + GAMMA*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[curr_state[0]][curr_state[1]])
            # if(Q>max_Q and not is_arr_in_list(next_state, visited)):
            if(Q>max_Q):
                max_Q = Q
                best_next_state = next_state
                best_state_success_prob = success_prob
                best_state_reward = reward
                best_action = action

        # rollout.append([curr_state,best_next_state,best_state_reward,best_state_success_prob,best_action]) #Use for reverse back-up

        tot_reward+=best_state_reward

        if(not max_Q>-float('Inf')):
            print("Going back to previous state")
            break

        """Un-comment the 4 lines below for a forward backup instead of a reverse one"""
        old_Q = V_table[curr_state[0]][curr_state[1]]
        V_table[curr_state[0]][curr_state[1]] = max_Q

        if(HAS_CONVERGED and abs(V_table[curr_state[0]][curr_state[1]]-old_Q)>ERROR_THRESHOLD):
            HAS_CONVERGED = False

        if np.random.rand() <= best_state_success_prob:
            potential_feasible_rollout.append([curr_state,best_next_state,best_action])
            # previous_state = curr_state
            # visited.append(best_next_state)
            curr_state = best_next_state

        if(np.array_equal(curr_state, goal_state)):
            print("Goal reached!!!!!")
            done = True
            tot_reward+=1000
            if(potential_reward<tot_reward):
                last_feasible_rollout = potential_feasible_rollout
                potential_reward = tot_reward
            REACHED_GOAL_ATLEAST_ONCE = True
            # HAS_CONVERGED = update_V_table(V_table,rollout,GAMMA,ERROR_THRESHOLD)
        episode_length+=1

    """Comment the 2 lines below for a forward backup instead of a reverse one"""
    # old_V_table = np.copy(V_table)
    # HAS_CONVERGED = update_V_table(V_table,rollout,GAMMA,ERROR_THRESHOLD)

    if(HAS_CONVERGED):
        print("Convergence of value function has been achieved after: ",num_episodes," episodes")
        break

print("Training complete!")
t1 = time.time()

total = t1-t0
print("Total training time is ",total," seconds")
# ipdb.set_trace()

env.s = start_state
done = False
total_reward = 0

if(not REACHED_GOAL_ATLEAST_ONCE and not HAS_CONVERGED):
    print("Algorithm never witnessed the goal state. Maybe no path to goal exists. Increase MAX_EPISODES or MAXIMUM_EPISODE_LENGTH and try again")
    exit()

elif(not HAS_CONVERGED):
    step_num = 0
    while(step_num<len(last_feasible_rollout) or not done):
        curr_state,forecasted_next_state,best_action = last_feasible_rollout[step_num]
        actual_next_state, reward, done, info = env.step(best_action) 
        total_reward+=reward
        if(np.array_equal(actual_next_state[:2], forecasted_next_state)):
            step_num+=1
        print("Next state is: ",actual_next_state[:2],"\t","Present_state is: ",curr_state,"\t Action is: ",best_action)

    if(done):
        print("Path to goal achieved successfully! Total reward is: ",total_reward)
    exit()
else:
    visited = [start_state]
    previous_state = None
    while not done:
        best_V_val = -float('Inf')
        curr_state = env.s
        for elt in range(len(dirX)):
            action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
            s_ = np.clip(env.s + np.array(action), 0, env.MAP_SIZE-1)
            if(V_table[s_[0]][s_[1]]>best_V_val and not is_arr_in_list(s_, visited)):
                best_action = action
                best_V_val = V_table[s_[0]][s_[1]]

        if(not best_V_val>-float('Inf')):
            print("Going back to previous state")
            env.s = previous_state 
        else:
            next_state, reward, done, info = env.step(best_action) 
            total_reward+=reward
            if(not is_arr_in_list(next_state[:2], visited)):
                visited.append(next_state[:2])
            if(not np.array_equal(next_state[:2], curr_state)):
                previous_state = curr_state
            print("Next state is: ",next_state[:2],"\t","Present_state is: ",curr_state,"\t Action is: ",best_action)

        if(done and np.array_equal(next_state[:2], goal_state)):
            print("Path to goal found. Goal reached successfully. Total reward is: ",total_reward)
        elif(done):
            print("Robot fell into water")
    exit()