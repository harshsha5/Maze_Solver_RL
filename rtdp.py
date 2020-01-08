import aid_gym
import numpy as np
import ipdb
import cv2

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

env = aid_gym.IslandEnv()
env.reset()

start_state = env.s
count = 0
while(True):
    env.s_goal = env.s + np.array([-20,20+count])
    if(not env._isValidPosition(env.s_goal)):
        count+=1
    else:
        break
goal_state = env.s_goal

display_map(start_state,goal_state)

print(env.MAP_SIZE)
print("start_state is: ",start_state,"\t","Goal State: ",goal_state)
ipdb.set_trace()
# V_table = np.zeros((env.MAP_SIZE, env.MAP_SIZE))
# obstacle_table = np.full((env.MAP_SIZE,env.MAP_SIZE), False, dtype=bool)

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
# obstacle_table = np.load('obstacle_table.npy')
V_table = np.load('initially_modified_v_table.npy')

V_table[goal_state[0]][goal_state[1]] = 1000

MAXIMUM_EPSIODE_LENGTH  = 10*int(np.linalg.norm(start_state-goal_state))
print("MAXIMUM_EPSIODE_LENGTH: ",MAXIMUM_EPSIODE_LENGTH)
MAX_EPISODES = 1000
HAS_CONVERGED = False
GAMMA = 1

dirX = [-1,-1,-1,0,0,1,1,1]
dirY = [-1,0,1,-1,1,-1,0,1]

curr_state = start_state
done = False

if(np.array_equal(start_state, goal_state)):
    print("Goal already reached")
    exit()

num_episodes=0
while(num_episodes<MAX_EPISODES):
    num_episodes+=1
    print("Episode: ",num_episodes)
    episode_length = 0
    curr_state = start_state
    done = False
    while(not done and episode_length<MAXIMUM_EPSIODE_LENGTH):
        max_Q = -float('Inf')
        for elt in range(len(dirX)):
            action = [dirX[elt],dirY[elt]]                    
            next_state = np.clip(curr_state + np.array(action), 0, env.MAP_SIZE-1)
            reward = -np.sqrt(np.square(action).sum())
            success_prob =  env._getSuccessProbability(env._getHeightDifference(curr_state, next_state[:2]))
            Q = reward + GAMMA*(success_prob*V_table[next_state[0]][next_state[1]] + (1-success_prob)*V_table[curr_state[0]][curr_state[1]])
            if(Q>max_Q):
                max_Q = Q
                best_next_state = next_state

        if(not max_Q>-float('Inf')):
            print("No path exists")
            break

        V_table[curr_state[0]][curr_state[1]] = max_Q

        if np.random.rand() <= success_prob:
            curr_state = best_next_state

        if(np.array_equal(curr_state, goal_state)):
            print("Goal reached!!!!!")
            done = True
        episode_length+=1

print("Training complete!")
ipdb.set_trace()

env.s = start_state
done = False
while not done:
    best_V_val = -float('Inf')
    curr_state = env.s
    for elt in range(len(dirX)):
        action = [dirX[elt],dirY[elt]]                       #See if we can get rid of actions which lead to out of the board configurations. Eg. the diagonal ones as you can simply move in one direction in that case. therefore using less fuel.
        s_ = np.clip(env.s + np.array(action), 0, env.MAP_SIZE-1)
        if(V_table[s_[0]][s_[1]]>best_V_val):
            best_action = action
            best_V_val = V_table[s_[0]][s_[1]]
        # ipdb.set_trace()

    if(not best_V_val>-float('Inf')):
        print("No Route to goal exists")
    else:
        next_state, reward, done, info = env.step(best_action) 
        print("Next state is: ",next_state[:2],"\t","Present_state is: ",curr_state,"\t Action is: ",best_action)

    if(done and np.array_equal(next_state[2:], goal_state)):
        print("Path to goal found. Goal reached successfully")
    elif(done):
        print("Robot fell into water")