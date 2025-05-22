import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import A2C

############### Environment configuration ###############
env = gym.make('merge-v0', render_mode='rgb_array')

# Uncomment for map visualization
'''
env.reset()
for _ in range(3):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()
'''

############### Training ###############
train = False
if train:
    model = A2C('MlpPolicy', env,
                learning_rate=5e-4,
                gamma=0.8,
                n_steps=5,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="new_a2c/",
                verbose=1)
    model.learn(total_timesteps=int(2e4))
    model.save("new_a2c/model")

############### Load and test saved model ###############
test = False
if test:
    model = A2C.load("new_a2c/model")
    for _ in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()

############### Test in episodes ###############
test_episode = True
if test_episode:
    model = A2C.load("new_a2c/model")
    crash_count = 0
    num_episodes = 25

    for _ in range(num_episodes):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            if 'crashed' in info and info['crashed']:
                crash_count += 1

    env.close()
    print(f"Total crashes in {num_episodes} episodes: {crash_count}, collision rate is {crash_count / num_episodes * 100:.2f}%")