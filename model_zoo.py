import torch
import torch.nn as nn
import numpy as np

class RNNModel_onset(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128):
    super().__init__()
    self.rnn = nn.GRU(num_nmels, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)
    
  def forward(self, x):
    mel_spec = x # batch size x num mels x time
    hidden_out, last_hidden = self.rnn(mel_spec.permute(0, 2, 1)) # batch size x time x hidden size *2
    logit = self.freq_projection(hidden_out) # batch size x time x 4
    prob = torch.sigmoid(logit)
    return prob.squeeze()
  
class RNNModel_onset_hfc_direct(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128, include_hfc=True):
    super().__init__()
    self.include_hfc = include_hfc
    input_size = num_nmels + 1 if include_hfc else num_nmels
    self.rnn = nn.GRU(input_size, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)

  def forward(self, x, hfc=None): # Include hfc as an optional input
    if self.include_hfc and hfc is not None:
      # Normalize the Mel spectrogram amplitudes (min-max normalization)
      #x_normalized = (x - x.min(dim=2, keepdim=True)[0]) / (x.max(dim=2, keepdim=True)[0] - x.min(dim=2, keepdim=True)[0] + 1e-6)
      x_normalized = x
      # Normalize the HFC values (min-max normalization)
      #hfc_normalized = (hfc - hfc.min(dim=1, keepdim=True)[0]) / (hfc.max(dim=1, keepdim=True)[0] - hfc.min(dim=1, keepdim=True)[0] + 1e-6)
      hfc_normalized = hfc
      # Concatenate the normalized Mel spectrogram and HFC values
      x = torch.cat((x_normalized, hfc_normalized.unsqueeze(1)), dim=1)
    
    hidden_out, last_hidden = self.rnn(x.permute(0, 2, 1)) # batch size x time x hidden size * 2
    logit = self.freq_projection(hidden_out) # batch size x time x 4
    prob = torch.sigmoid(logit)
    return prob.squeeze()
  
class RNNModel_onset_hfc_indirect(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128):
    super().__init__()
    self.rnn_mel_spec = nn.GRU(num_nmels, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.rnn_hfc = nn.GRU(1, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.merge_layer = nn.Linear(hidden_size * 4, hidden_size * 2)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)

  def forward(self, mel_spec, hfc):
    hfc_expanded = hfc.unsqueeze(1) # batch size x 1 x time

    # Mel spectrogram branch
    hidden_out_mel, _ = self.rnn_mel_spec(mel_spec.permute(0, 2, 1)) # batch size x time x hidden size * 2

    # HFC branch
    hidden_out_hfc, _ = self.rnn_hfc(hfc_expanded.permute(0, 2, 1)) # batch size x time x hidden size * 2
    
    # Merge the two branches
    combined = torch.cat((hidden_out_mel, hidden_out_hfc), dim=2) # batch size x time x (hidden size * 2 * 2)
    merged = self.merge_layer(combined)  # batch size x time x hidden size * 2

    logit = self.freq_projection(merged)   # batch size x time x 4
    prob = torch.sigmoid(logit)

    return prob.squeeze()

class RNNModel_velocity(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128):
    super().__init__()
    self.rnn = nn.GRU(num_nmels, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)
    
  def forward(self, x, target_drum_roll):
    mel_spec = x # batch size x num mels x time
    hidden_out, last_hidden = self.rnn(mel_spec.permute(0, 2, 1)) # batch size x time x hidden size *2
    logit = self.freq_projection(hidden_out) # batch size x time x 4
    logit = torch.einsum('bij,bij->bij', logit, target_drum_roll)
    prob = torch.sigmoid(logit)
    return prob.squeeze()
  
class RNNModel_velocity_hfc_directly(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128, include_hfc=True):
    super().__init__()
    self.include_hfc = include_hfc
    input_size = num_nmels + 1 if include_hfc else num_nmels
    self.rnn = nn.GRU(input_size, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)
  
  def forward(self, x, target_drum_roll, hfc=None): # Include hfc as an optional input
    if self.include_hfc and hfc is not None:
      # Normalize the Mel spectrogram amplitudes (min-max normalization)
      #x_normalized = (x - x.min(dim=2, keepdim=True)[0]) / (x.max(dim=2, keepdim=True)[0] - x.min(dim=2, keepdim=True)[0] + 1e-6)
      x_normalized = x
      # Normalize the HFC values (min-max normalization)
      #hfc_normalized = (hfc - hfc.min(dim=1, keepdim=True)[0]) / (hfc.max(dim=1, keepdim=True)[0] - hfc.min(dim=1, keepdim=True)[0] + 1e-6)
      hfc_normalized = hfc
      # Concatenate the normalized Mel spectrogram and HFC values
      x = torch.cat((x_normalized, hfc_normalized.unsqueeze(1)), dim=1)

    hidden_out, last_hidden = self.rnn(x.permute(0, 2, 1)) # batch size x time x hidden size * 2
    logit = self.freq_projection(hidden_out) # batch size x time x 4
    logit = torch.einsum('bij,bij->bij', logit, target_drum_roll)
    prob = torch.sigmoid(logit)
    return prob.squeeze()
  
class RNNModel_velocity_hfc_indirect(nn.Module):
  def __init__(self, num_nmels=128, hidden_size=128):
    super().__init__()
    self.rnn_mel_spec = nn.GRU(num_nmels, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.rnn_hfc = nn.GRU(1, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
    self.merge_layer = nn.Linear(hidden_size * 4, hidden_size * 2)
    self.freq_projection = nn.Linear(hidden_size * 2, 4)

  def forward(self, mel_spec, target_drum_roll, hfc):
    hfc_expanded = hfc.unsqueeze(1) # batch size x 1 x time

    # Mel spectrogram branch
    hidden_out_mel, _ = self.rnn_mel_spec(mel_spec.permute(0, 2, 1)) # batch size x time x hidden size * 2

    # HFC branch
    hidden_out_hfc, _ = self.rnn_hfc(hfc_expanded.permute(0, 2, 1)) # batch size x time x hidden size * 2
    
    # Merge the two branches
    combined = torch.cat((hidden_out_mel, hidden_out_hfc), dim=2) # batch size x time x (hidden size * 2 * 2)
    merged = self.merge_layer(combined)  # batch size x time x hidden size * 2

    logit = self.freq_projection(merged)   # batch size x time x 4
    logit = torch.einsum('bij,bij->bij', logit, target_drum_roll)
    prob = torch.sigmoid(logit)

    return prob.squeeze()

class Linear_velocity(nn.Module):
  def __init__(self):
    super().__init__()
    self.freq_projection = nn.Linear(4, 4)
    self.freq_projection2 = nn.Linear(4, 4)
    self.relu = nn.ReLU()
    
  def forward(self, target_drum_roll):
    logit = self.freq_projection(target_drum_roll)
    logit = self.relu(logit)
    logit = self.freq_projection2(logit)
    prob = torch.sigmoid(logit) # batch size x time x 4
    return prob.squeeze()

