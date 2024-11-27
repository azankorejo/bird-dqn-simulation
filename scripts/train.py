import os
import torch

def train_dqn(agent, env, model_save_path="trained_model.pth", episodes=1000, save_every_n_episodes=100):
    # Check if a trained model already exists
    if os.path.exists(model_save_path):
        print("Found a trained model, loading...")
        # Load the saved model weights into the Q-network
        agent.q_network.load_state_dict(torch.load(model_save_path))
        agent.target_network.load_state_dict(torch.load(model_save_path))  # If you have a target network
        agent.q_network.eval()  # Set the Q-network to evaluation mode (disable dropout, batch norm)
    else:
        print("No trained model found, starting training from scratch...")
    
    # Start training loop
    for episode in range(episodes):
        state = env.get_state()  # Get the initial state
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)  # Select action based on current policy
            next_state, reward, done = env.step(action)  # Take the action and get the next state and reward
            agent.store_experience(state, action, reward, next_state, done)  # Store the experience
            agent.train()  # Train the agent on the experience

            state = next_state  # Update state
            total_reward += reward  # Update total reward
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        # Save the model after every `save_every_n_episodes`
        if (episode + 1) % save_every_n_episodes == 0:
            print(f"Saving model at episode {episode + 1}...")
            torch.save(agent.q_network.state_dict(), model_save_path)
            torch.save(agent.target_network.state_dict(), model_save_path)  # Save target network if you have one

    print("Training complete!")
