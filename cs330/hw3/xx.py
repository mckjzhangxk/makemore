import gym
env=gym.make("FrozenLake-v1",render_mode='human',is_slippery=False)
env.reset(seed=22)

for i in range(10):
    next_state,reward,end,_,info=env.step(2)
    print()
    env.render()