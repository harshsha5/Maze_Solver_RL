import aid_gym

env = aid_gym.IslandEnv()
env.reset()

total_reward = 0
for i in range(1000):
    a = env.actionSpace.sample()
    state, reward, done, info = env.step(a)
    total_reward += reward
    print(i, state, reward)
    if done:
        break

print(state, total_reward)
