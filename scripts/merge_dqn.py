import gymnasium as gym
from matplotlib import pyplot as plt
#import pprint
from stable_baselines3 import DQN

############### Environment configuration ###############
env = gym.make('merge-v0', render_mode='rgb_array')

# Display environment coniguration
#pprint.pprint(env.config)
'''  # comment out for map visualization
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
if train == True:
  model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="new1201_dqn/")
  model.learn(int(2e4))
  model.save("new1201_dqn/model")


############### Load and test saved model ###############
test = False
if test == True:
  model = DQN.load("new1201_dqn/model")
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
if test_episode == True:
  model = DQN.load("new1201_dqn/model")
  # Set test to True to run this block
  test = True
  if test:
      crash_count = 0
      success_merge = 0
      num_episodes = 25

      for _ in range(num_episodes):
          done = truncated = False
          obs, info = env.reset()
          #print(info)
          while not (done or truncated):
              action, _states = model.predict(obs, deterministic=True)
              obs, reward, done, truncated, info = env.step(action)
              '''
              if 'action' in info and 'action' == 1:
                success_merge += 1
              '''
              env.render()

              # Check for crashes
              if 'crashed' in info and info['crashed']:
                  crash_count += 1

      env.close()

      print(f"Total crashes in {num_episodes} episodes: {crash_count}, collision rate is {crash_count/num_episodes*100}%")
      #print(f"Total crashes in {num_episodes} episodes: {success_merge}, success rate is {success_merge/num_episodes*100}%")