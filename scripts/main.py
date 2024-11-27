from bird import HummingbirdSimulation
from control import HummingbirdControl
import os
from train import train_dqn
from models.agent import DQNAgent

# Initialize the simulation environment
env = HummingbirdSimulation(
    model_path="../3d/bird.urdf",
    texture_path="../3d/Kolibri_baseColor.png",
    plane_texture_path="../3d/grass.jpg"
)
# Initialize the agent
state_dim = 6  # Example: [x, y, z, vx, vy, vz]
action_dim = 6  # Example: [forward, backward, left, right, up, down]
agent = DQNAgent(state_dim, action_dim)

# Path to save the trained model
model_save_path = "trained_model.pth"

# Train or continue training the agent
train_dqn(agent, env, model_save_path=model_save_path, episodes=10000, save_every_n_episodes=1000)
