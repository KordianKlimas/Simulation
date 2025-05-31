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

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def save_gru_model(model, path="feature_gru_model.pth"):
    torch.save(model.state_dict(), path)

def load_gru_model(path, input_size, hidden_size, num_layers, num_outputs, device=None):
    model = GRU(input_size, hidden_size, num_layers, num_outputs)
    if device:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
    else:
        model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_with_gru(model, feature_sequence, device=None):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(feature_sequence, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_size)
        if device:
            x = x.to(device)
        output = model(x)
        return output.cpu().numpy().squeeze()