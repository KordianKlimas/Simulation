# Python
import torch

class Trainer:
    def __init__(self, env, model, ppo_trainer, rollout_buffer, max_iterations=1000, max_steps=500):
        self.env = env
        self.model = model
        self.ppo_trainer = ppo_trainer
        self.buffer = rollout_buffer
        self.max_iterations = max_iterations
        self.max_steps = max_steps

    # Python
    def policy(self, state_dict):
        """
        Determines the action based on the current state dictionary.
        """

        # Flatten nested dictionaries
        def flatten_dict(d, parent_key='', sep='_'):
            if not isinstance(d, dict):
                raise TypeError(f"Expected a dictionary, but got {type(d)}")
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    # Recursively flatten nested dictionaries
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Handle lists by processing each element
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            # Recursively flatten dictionaries inside lists
                            items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                        elif isinstance(item, (int, float)):
                            items.append((f"{new_key}{sep}{i}", item))
                        elif hasattr(item, '__dict__'):
                            # Flatten custom objects by extracting their attributes
                            items.extend(flatten_dict(item.__dict__, f"{new_key}{sep}{i}", sep=sep).items())
                        else:
                            raise TypeError(f"Unsupported value type in list: {type(item)} for key: {new_key}{sep}{i}")
                elif isinstance(v, (int, float)):  # Only include real numbers
                    items.append((new_key, v))
                elif hasattr(v, '__dict__'):
                    # Flatten custom objects by extracting their attributes
                    items.extend(flatten_dict(v.__dict__, new_key, sep=sep).items())
                else:
                    raise TypeError(f"Unsupported value type: {type(v)} for key: {new_key}")
            return dict(items)

        # Flatten the state dictionary
        flat_state_dict = flatten_dict(state_dict)

        # Convert flattened dictionary values to a tensor
        state_tensor = torch.tensor(list(flat_state_dict.values()), dtype=torch.float32).unsqueeze(0)
        
        # Verify the shape of the state tensor
        if state_tensor.shape[1] != self.model.actor.in_features:
            raise ValueError(f"State tensor shape mismatch: expected {self.model.actor.in_features}, got {state_tensor.shape[1]}")
        
        with torch.no_grad():
            action_probs = self.model(state_tensor)  # Forward pass through the model
        action_distribution = torch.distributions.Categorical(logits=action_probs)
        action = action_distribution.sample().item()  # Sample an action
        return action

    def run_episode(self, initial_state):
        # Ensure the initial_state is a dictionary and flatten it
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, (int, float)):
                    items.append((new_key, v))
            return dict(items)

        if isinstance(initial_state, dict):
            initial_state = flatten_dict(initial_state)
        elif isinstance(initial_state, (list, tuple)):
            initial_state = {f"feature_{i}": value for i, value in enumerate(initial_state)}

        state = initial_state  # Use the provided initial state
        done = False
        total_reward = 0

        while not done:
            state = flatten_dict(state)  # Ensure state is flattened before passing to policy
            action = self.policy(state)  # Get action from policy
            next_state, reward, done, *_ = self.env.process_turn()  # Step in environment

            # Ensure next_state is a dictionary and flatten it
            if isinstance(next_state, dict):
                next_state = flatten_dict(next_state)
            elif isinstance(next_state, (list, tuple)):
                next_state = {f"feature_{i}": value for i, value in enumerate(next_state)}

            self.update(state, action, reward, next_state, done)  # Update model
            state = next_state
            total_reward += reward

        return total_reward

    def update(self, state, action, reward, next_state, done):
        # Placeholder for update logic
        pass

    def train(self, initial_state):
        """
        Executes the training loop for a specified number of iterations.
        """
        for iteration in range(self.max_iterations):
            total_reward = self.run_episode(initial_state)
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Total Reward: {total_reward}")





