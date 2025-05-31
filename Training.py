# NOTE
# Many sequence-to-vector task model 
# Takes in a sequence of feature vectors every n ticks and outputs a single vector
# that represents the desired state of the game. 
# Input: feature vectors fot each tick in some interval (e.g. 100 ticks - 10sec) - sequence_length
# Output variables:
# desired_player_hp_change_rate
# desired_played_dodge_rate
# desired_player_damage_output_rate_to_boss
# desired_time_since_boss_last_took_damage_from_player

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from AI.GRU import GRU, save_gru_model
from Test import it_settings, device  # Or import from a config module

GRU_settings = it_settings["GRU_settings"]

input_size = GRU_settings["input_size"]
sequence_length = GRU_settings["sequence_length"]
hidden_size = GRU_settings["hidden_size"]
num_layers = GRU_settings["num_layers"]
num_outputs = GRU_settings["num_outputs"]

num_epochs = GRU_settings["num_epochs"]
batch_size = GRU_settings["batch_size"]
learning_rate = GRU_settings["learning_rate"]

# Example: Dummy dataset creation (replace with your real feature vector data)
# Generate random data: (num_samples, sequence_length, input_size)
num_samples = 1000
X = np.random.randn(num_samples, sequence_length, input_size).astype(np.float32)
y = np.random.randn(num_samples, num_outputs).astype(np.float32)  # Targets

# Convert to torch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Create TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = GRU(input_size, hidden_size, num_layers, num_outputs).to(device)

# Loss and optimizer (MSE for regression, change if you have classification)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (features, targets) in enumerate(train_loader):
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the model
save_gru_model(model, GRU_settings["model_path"])