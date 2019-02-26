import gym

env = gym.make('CarRacing-v0')
observations = env.reset()

print(env.action_space)
print(env.observation_space)

env.render("human")

rew_tot = 0
obs = env.reset()
env.render("human")
for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    rew_tot = rew_tot + rew
    env.render("human")

print(f"Reward: {rew_tot}")
