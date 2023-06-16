import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
import IPython.display as ipd
import torch.nn as nn 
#from torch.optim.lr_scheduler import StepLR



class drum_generation_model(nn.Module):
  #def __init__(self, x_emb_size = 8, y_emb_size = 16, encoder_hidden_size = 32, decoder_hidden_size = 128 , style_len = 18):
  def __init__(self, x_emb_size = 32, y_emb_size = 64, encoder_hidden_size = 64, decoder_hidden_size = 512 , style_len = 18):
    super().__init__() 
    self.x_emb = nn.Sequential(nn.Linear(4, x_emb_size), nn.LayerNorm(x_emb_size), nn.ReLU())
    self.encoder = nn.GRU(x_emb_size, encoder_hidden_size, bidirectional = True, batch_first = True, num_layers = 2, dropout = 0.4)
    
    self.start_token_emb = nn.Embedding(style_len, y_emb_size)
    self.start_token_lin = nn.Linear(y_emb_size, decoder_hidden_size * 3)
  
    self.vel_decoder = nn.GRU(y_emb_size + encoder_hidden_size*2, decoder_hidden_size, batch_first = True, num_layers = 3, dropout = 0.4)
    
    #self.onset_decoder = nn.GRU(y_emb_size + encoder_hidden_size*2 , decoder_hidden_size, batch_first = True, num_layers=3)
    
    #self.onset_y_emb = nn.Sequential(nn.Linear(9, y_emb_size), nn.ReLU())
    self.vel_y_emb = nn.Sequential(nn.Linear(9, y_emb_size), nn.LayerNorm(y_emb_size),  nn.ReLU())
    
    self.vel_proj = nn.Sequential(nn.Linear(decoder_hidden_size, 9), nn.LayerNorm(9), nn.Sigmoid())
    #self.onset_proj = nn.Sequential(nn.Linear(decoder_hidden_size, 9), nn.Sigmoid())
    
      
    self.vel_cnn = nn.Sequential(nn.Conv1d(9, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                                   nn.Conv1d(64, 9, 3, padding=1), nn.Sigmoid())
    
    self.onset_cnn = nn.Sequential(nn.Conv1d(9, 64, 3, padding=1), nn.LayerNorm(64), nn.ReLU(),
                                   nn.Conv1d(64, 9, 3, padding=1), nn.Sigmoid())
      # 64, 128), nn.ReLu(), nn.Linear(128, 64), nn.Sigmoid())
    # self.final_onset = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.Sigmoid())
    
  def forward(self, x, style_idx, y = None, teacher_forcing_ratio=1.0): # teacher_forcing_ratio = 1.0):
    x = self.x_emb(x) # [batch, 64, 4] -> [batch, 64, 16] -> input only velocity
    enc_hidden, enc_last_hidden = self.encoder(x) # [batch, 64, emb_size(16)] -> [batch, 64, encoder_hidden_size(16) * 2]
    
    
    start_token = self.start_token_emb(style_idx) #[batch, 1, ]

    decoder_last = self.start_token_lin(start_token)
    decoder_last = decoder_last.chunk(3, dim = -1)
    decoder_last = torch.stack(decoder_last, dim = 0)
    val_decoder_last = decoder_last    
    
    vel_decoder_input = torch.cat([enc_hidden[:,0,:], start_token], dim = 1).unsqueeze(1) #16 + 16 + 9 

            
    outputs = []
    onset_outputs = []
      
    
    if y != None:
      vel_y = self.vel_y_emb(y)
      
      vel_concat_vector = torch.cat([enc_hidden[:,1:,:], vel_y[:, :-1, :]], dim = -1) #[256, 63, 25]
      
      vel_concat_vector = torch.cat([vel_decoder_input, vel_concat_vector], dim = 1)
      
      # vel_decoder_hidden, val_decoder_last = self.vel_decoder(vel_concat_vector, val_decoder_last)
            
      # outputs = self.vel_proj(vel_decoder_hidden)

    for time_index in range(0, 64):
      
      vel_decoder_hidden, val_decoder_last = self.vel_decoder(vel_decoder_input, val_decoder_last)
      
      vel_out = self.vel_proj(vel_decoder_hidden)

      outputs.append(vel_out)
      
      # vel_out_probs = vel_out * (onset_out > 0.5).float()
      # vel_out_probs = vel_out_probs.permute(0, 2, 1)
      
      # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
      
      # if teacher_forcing_ratio == 0:
      #   use_teacher_forcing = False
      use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

      if teacher_forcing_ratio == 0:
        use_teacher_forcing = False
        
      if time_index <= 62: 
        if use_teacher_forcing:
          vel_decoder_input = vel_concat_vector[:, time_index, :].unsqueeze(1)
        else:
          vel_decoder_input = torch.cat([enc_hidden[:, time_index+1, :].unsqueeze(1), self.vel_y_emb(vel_out)], dim = -1)
      
    outputs = torch.cat(outputs, dim = 1) #[batch, 64, 9] -> cat
      
    outputs = outputs.permute(0, 2, 1) #[batch, 9, 64]
    
    onset_outputs = self.onset_cnn(outputs)
    outputs = self.vel_cnn(outputs)
    
    return outputs, onset_outputs
  
  