import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 1024),  # Use input_dim dynamically
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.actor = nn.Linear(512, output_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, state_dict):
        """
        Forward pass for the ActorCritic model.
        """

        # Ensure state_dict is flattened and contains only numeric values
        # Python
        def preprocess_state(state_dict):
            """
            Preprocess the state_dict to ensure it is flattened and contains only numeric values.
            """
            if isinstance(state_dict, dict):
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        elif isinstance(v, (int, float)):
                            items.append((new_key, v))
                        else:
                            raise ValueError(f"Unsupported value type: {type(v)} for key: {new_key}")
                    return dict(items)

                flat_state_dict = flatten_dict(state_dict)
                return list(flat_state_dict.values())
            elif isinstance(state_dict, torch.Tensor):
                # If already a tensor, convert to a list of values
                return state_dict.flatten().tolist()
            else:
                raise TypeError(f"Unsupported state_dict type: {type(state_dict)}")
            return list(flat_state_dict.values())

        # Preprocess the state_dict
        flat_state_values = preprocess_state(state_dict)

        # Convert to tensor
        state_tensor = torch.tensor(flat_state_values, dtype=torch.float32).unsqueeze(0)

        # Ensure the input tensor has the correct shape
        if state_tensor.shape[1] != self.actor.in_features:
            raise ValueError(f"Expected input features {self.actor.in_features}, but got {state_tensor.shape[1]}")

        # Forward pass
        logits = self.actor(state_tensor)
        value = self.critic(state_tensor)
        return logits, value

class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

class PPOTrainer:
    def __init__(self, model, learning_rate=3e-4, gamma=0.99, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute_returns(self, rewards, dones, values, last_value=0.0):
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, buffer):
        states = torch.stack(buffer.observations)
        actions = torch.tensor(buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
        returns = torch.tensor(self.compute_returns(buffer.rewards, buffer.dones, buffer.values), dtype=torch.float32)

        for _ in range(4):
            probs, values = self.model(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            advantages = returns - values.squeeze().detach()

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = (returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

