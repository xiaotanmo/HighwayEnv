import gymnasium as gym
#import highway_env
from matplotlib import pyplot as plt

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}
env = gym.make('merge-v0')
env.configure(config)
obs, info = env.reset()



config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (1200, 200),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2
}
env.configure(config)
obs, info = env.reset()
'''
fig, axes = plt.subplots(ncols=4, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
plt.show()
'''
# Create a single plot for the fourth figure
fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(obs[3, ...].T, cmap=plt.get_cmap('gray'))  # Index 3 for the fourth element
ax.set_xlim(300, 1100)

plt.show()