import random
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.8, epsilon=0.8, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Check if CUDA (GPU) is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the Q-network and move it to the appropriate device
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.099)
        
        self.memory = []
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # Exploration
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values, dim=1).item()  # Exploitation
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a mini-batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        # Compute the Q-values for the current states
        q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        # Compute the target Q-values (Bellman equation)
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
        
        # Compute loss and optimize the Q-network
        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
