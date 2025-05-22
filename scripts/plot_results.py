import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

# Function to extract data from TensorBoard logs
def extract_tensorboard_data(logdir, tag):
    steps = []
    values = []
    for file in os.listdir(logdir):
        if file.startswith("events"):
            for e in summary_iterator(os.path.join(logdir, file)):
                for v in e.summary.value:
                    if v.tag == tag:
                        steps.append(e.step)
                        values.append(v.simple_value)
    return steps, values

logdir = 'new1201_dqn/DQN_1'  # Adjust to log directory

# Extracting different metrics from the logs
learning_rate_steps, learning_rate = extract_tensorboard_data(logdir, 'train/learning_rate')
loss_steps, loss = extract_tensorboard_data(logdir, 'train/loss')
exploration_rate_steps, exploration_rate = extract_tensorboard_data(logdir, 'rollout/exploration_rate')
rewards_steps, rewards = extract_tensorboard_data(logdir, 'rollout/ep_rew_mean')

# Creating subplots
fig, axs = plt.subplots(1, 4, figsize=(20, 4))

# Plot Learning Rate Over Time
axs[0].plot(learning_rate_steps, learning_rate)
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Learning Rate')
axs[0].set_title('Learning Rate Over Time')
axs[0].grid(True)

# Plot Training Loss Over Time
axs[1].plot(loss_steps, loss)
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Loss')
axs[1].set_title('Training Loss Over Time')
axs[1].grid(True)

# Plot Exploration Rate Over Time
axs[2].plot(exploration_rate_steps, exploration_rate)
axs[2].set_xlabel('Steps')
axs[2].set_ylabel('Exploration Rate')
axs[2].set_title('Exploration Rate Over Time')
axs[2].grid(True)

# Plot Training Rewards Over Time
axs[3].plot(rewards_steps, rewards)
axs[3].set_xlabel('Steps')
axs[3].set_ylabel('Average Reward')
axs[3].set_title('Training Rewards Over Time')
axs[3].grid(True)

# Display the plots
plt.tight_layout()
plt.show()
