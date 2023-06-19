import torch
import torchaudio
import numpy as np
from pathlib import Path
import IPython.display as ipd
import torch.nn as nn 
import random 
#from torch.optim.lr_scheduler import StepLR

class drum_encoder_model(nn.Module):
  def __init__(self):
    super().__init__() 
    self.encoder_emb = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
    self.encoder = nn.GRU(32, 128, bidirectional = True, batch_first = True, num_layers = 3, dropout = 0.4)
    self.encoder_proj = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(128, 64), nn.ReLU(),  nn.Dropout(0.2),
                                      nn.Linear(64, 32), nn.ReLU(),
                                      nn.Linear(32, 4), nn.Sigmoid())
  def forward(self, x):
    x = self.encoder_emb(x)
    x, _ = self.encoder(x)
    x = self.encoder_proj(x)
    return x
  
class drum_decoder_model(nn.Module):
  def __init__(self):
    super().__init__() 
    self.decoder_hidden_size = 1024
    self.decoder = nn.GRU(256, self.decoder_hidden_size, bidirectional = False, batch_first = True, num_layers = 3, dropout = 0.4)
    self.decoder_proj = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(64, 9), nn.Sigmoid())
  def forward(self, x, decoder_last):
    x, decoder_last = self.decoder(x, decoder_last)
    x = self.decoder_proj(x)
    return x, decoder_last
      
class drum_gen_model(nn.Module):
  def __init__(self):
    super().__init__() 
    self.encoder = drum_encoder_model()
    self.decoder = drum_decoder_model()
    self.encoder_emb = nn.Sequential(nn.Linear(4, 128), nn.ReLU())
    self.y_emb = nn.Sequential(nn.Linear(9, 128), nn.ReLU())
    self.start_token_emb = nn.Embedding(18, 128)
    self.start_token_lin = nn.Sequential(nn.Linear(128, self.decoder.decoder_hidden_size * 3), nn.ReLU(), nn.Dropout(0.2))
    self.onset_cnn = nn.Sequential(nn.Conv1d(9, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
                                   nn.Conv1d(32, 9, 3, padding=1), nn.Sigmoid())
                                   
  def forward(self, x, style_idx, y = None, validation_mode = False, teacher_forcing_ratio = 1.0):
    enc_x = self.encoder(x) #[batch, 64, 4]
    x = self.encoder_emb(enc_x) #[batch, 64, 128]
    
    start_token = self.start_token_emb(style_idx) #[batch, y_emb_size]

    decoder_last = self.start_token_lin(start_token)
    decoder_last = decoder_last.chunk(3, dim = -1)
    
    decoder_last = torch.stack(decoder_last, dim = 0)
  
    # if y != None:
    #   y = self.y_emb(y[:, :-1, :]) # [batch, 64, 128]
    #   teacher_start_token = torch.cat([start_token.unsqueeze(1), y], dim = 1)
    #   teacher_start_token = torch.cat([x, start_token], dim = -1) #[batch, 64, 256]
    
    velocity_pred = [] 
    
    for time_index in range(0, 64):
      input_token = torch.cat([x[:, time_index, :], start_token], dim=-1).unsqueeze(1)
      vel_pred, decoder_last = self.decoder(input_token, decoder_last)
      velocity_pred.append(vel_pred)
      
      use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
      
      if use_teacher_forcing:
        start_token = self.y_emb(y[:, time_index, :])
      else:
        start_token = self.y_emb(vel_pred).squeeze(1)
      
      vel_pred = torch.cat(velocity_pred, dim = 1) #[batch, 64, 9] -> cat
    
      
      

    # if validation_mode == True:
    #   velocity_pred = []
    #   onset_pred = []
    #   for time_index in range(0, 64):
    #     input_token = torch.cat([x[:, time_index, :], start_token], dim=-1).unsqueeze(1)
    #     vel_pred = self.decoder(input_token, decoder_last)
    #     velocity_pred.append(vel_pred)
    #     vel_pred = vel_pred * 127
    #     vel_pred = torch.round(vel_pred)
    #     vel_pred = vel_pred / 127
    #     start_token = self.y_emb(vel_pred).squeeze(1)
        
    #   vel_pred = torch.cat(velocity_pred, dim = 1) #[batch, 64, 9] -> cat

        
    # else:
    #   y = self.y_emb(y[:, :-1, :]) # [batch, 64, 128]
    #   start_token = torch.cat([start_token.unsqueeze(1), y], dim = 1)
    #   start_token = torch.cat([x, start_token], dim = -1) #[batch, 64, 256]
    #   vel_pred = self.decoder(start_token, decoder_last)

    
    vel_pred = vel_pred.permute(0, 2, 1)
    onset_pred = self.onset_cnn(vel_pred)
    
    return vel_pred, onset_pred
  
  