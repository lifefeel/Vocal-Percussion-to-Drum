import torch
import torchaudio

import numpy as np
import librosa
import matplotlib.pyplot as plt

import pickle
import argparse
from pathlib import Path
import IPython.display as ipd
from infer_util import get_nine_midi, plot_pianoroll
from model import drum_generation_model

from model_zoo import RNNModel_onset, RNNModel_velocity

class SpecConverter():
  def __init__(self, sr=44100, n_fft=2048, hop_length=1024, n_mels=128, fmin=0, fmax=None):
    self.sr = sr
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.fmin = fmin
    self.fmax = fmax
  
  def forward(self, wav):
    mel_spec = torchaudio.transforms.MelSpectrogram(
      sample_rate=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec(wav))
    return log_mel_spec
  
def high_freq_content(spectrogram_dB):
  # Convert the decibel values back to linear amplitude values
  spectrogram = librosa.db_to_power(spectrogram_dB)

  # Create a frequency axis
  freqs = librosa.core.mel_frequencies(n_mels=128)

  # Calculate the weighted mean of the amplitude for each bin
  hfc_values = np.empty(spectrogram.shape[1])
  for t in range(spectrogram.shape[1]):
    hfc_t = np.sum(freqs * spectrogram[:, t])
    hfc_values[t] = hfc_t

  return hfc_values

def reducing_time_resolution(mel_spec, aggregate_factor=4, len_quantized=16):
  db_mel_spec_cnvtd = []
  for idx in range(len_quantized):
    spec_for_agg = mel_spec[:, idx*aggregate_factor:(idx+1)*aggregate_factor]
    aggregated_spec = torch.mean(spec_for_agg, dim=1, keepdim=True)
    #print(aggregated_spec[0].shape)
    db_mel_spec_cnvtd.append(aggregated_spec)
  db_mel_spec_cnvtd = torch.cat(db_mel_spec_cnvtd, dim=1)
  return db_mel_spec_cnvtd

def denoise(drum_roll):
  first_onset_count = 90
  for row in drum_roll:
    for idx, val in enumerate(row):
      if val > 0:
        if first_onset_count - idx > 0:
          first_onset_count = idx
          break
      elif first_onset_count - idx < 0:
        break
  
  return drum_roll[:, first_onset_count:]

def Dense_onsets(tensor):
  result = np.zeros_like(tensor)
  avg = 0
  count = 0
  first_non_zero_index = None

  for i, value in enumerate(tensor):
    if value != 0:
      if first_non_zero_index is None:
        first_non_zero_index = i
      avg += value
      count += 1
    else:
        if count != 0:
          result[first_non_zero_index] = avg / count
          avg, count, first_non_zero_index = 0, 0, None
  if count != 0:
    result[first_non_zero_index] = avg / count

  return result

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--wav_path', type=str, default='audio_samples/pmta.wav')
  return parser

def plt_imshow(npimg, title=None, filename=None):
  plt.figure(figsize=(20, 10))
  plt.imshow(npimg, aspect='auto', origin='lower', interpolation='nearest')
  if title is not None:
    plt.title(title)
    output_path = f'transcribed_sample_results/{title}_{filename}.png'
    plt.savefig(output_path)
    print('saved:', output_path)
    
  

if __name__ == "__main__":
  args = get_argument_parser().parse_args()
  print('input audio:', args.wav_path)
  audio_path = Path(args.wav_path)
  sample_first, sr = torchaudio.load(audio_path)

  device = torch.device('cpu')
  onset_model = RNNModel_onset(num_nmels=128, hidden_size=128).to(device)
  velocity_model = RNNModel_velocity(num_nmels=128, hidden_size=128).to(device)

  # load model
  onset_model.load_state_dict(torch.load('models/onset_model_noQZ.pt'))
  velocity_model.load_state_dict(torch.load('models/velocity_model_noQZ.pt'))

  spec_converter = SpecConverter(sr=44100, n_fft=512, hop_length=128, n_mels=128)

  mel_spec = spec_converter.forward(sample_first.unsqueeze(0))
  mel_spec = mel_spec[0][0][:,:2756]
  mel_spec = mel_spec.to(device)

  threshold = 0.4
  onset_pred = onset_model(mel_spec.unsqueeze(0))
  onset_pred_guide = (onset_pred > threshold).float() # time x 4
  velocity_pred = velocity_model(mel_spec.unsqueeze(0), onset_pred_guide.unsqueeze(0))
  velocity_pred = velocity_pred * onset_pred_guide.unsqueeze(0)

  hfc_values = high_freq_content(mel_spec.cpu().detach().numpy())

  onset_idx = np.argwhere(hfc_values > np.percentile(hfc_values, 60))
  onset_pred_cleaned = torch.zeros_like(velocity_pred.squeeze())
  for idx in onset_idx.squeeze():
    onset_pred_cleaned[idx] = velocity_pred.squeeze()[idx]

  plt_imshow(onset_pred_cleaned.cpu().detach().numpy().T, title='onset_pred_cleaned', filename=audio_path.stem)

  aggregate_factor = onset_pred_cleaned.shape[0] // 128
  db_mel_spec_cnvtd = reducing_time_resolution(onset_pred_cleaned.T, aggregate_factor, 128) # 128 x timestep

  threshold_idx = torch.argwhere(db_mel_spec_cnvtd > 0.3)
  drum_roll_QZ = torch.zeros_like(db_mel_spec_cnvtd)
  for row in threshold_idx:
    drum_roll_QZ[row[0], row[1]] = db_mel_spec_cnvtd[row[0], row[1]]

  denoised_drum_roll = denoise(drum_roll_QZ)

  densed_drumroll = np.zeros_like(denoised_drum_roll.cpu().detach().numpy())
  for idx, row in enumerate(denoised_drum_roll.cpu().detach().numpy()):
    densed_drumroll[idx] = Dense_onsets(row)

  plt_imshow(densed_drumroll, title='densed_drumroll', filename=audio_path.stem)

  with open(f'transcribed_sample_results/{audio_path.stem}.pkl', 'wb') as f:
    pickle.dump(densed_drumroll[:,:64], f)
    
  x = torch.from_numpy(densed_drumroll[:,:64]).unsqueeze(0)
    
  # bpm = args.bpm
  # style = args.style
  bpm = 120
  style = 'rock'
  
  style2idx = {'gospel': 0, 'blues': 1,'afrobeat': 2,'jazz': 3,'country': 4,'funk': 5,'rock': 6,
                      'hiphop': 7,'latin': 8,'dance': 9,'middleeastern': 10,'reggae': 11,'neworleans': 12,
                      'punk': 13,'pop': 14,'soul': 15,'highlife': 16,'afrocuban': 17}

  
  # style_idx = torch.tensor(style2idx[style], dtype=torch.long)
  style_idx = torch.tensor([style2idx[style]], dtype=torch.long)

  model = drum_generation_model()
  model.load_state_dict(torch.load('./models/17model_100.pt')) # 아직 가중치가 없어서 추가할 예정입니다. 모델구조와 가중치는 따로 수정할 수 있도록 해주시면 감사하겠습니다! 
  
  model.eval()
  with torch.no_grad():
    x_reshape = x.permute(0, 2, 1)
    vel_outputs, onset_outputs = model(x = x_reshape, style_idx= style_idx, teacher_forcing_ratio = 0.0) 
    # input이 들어와야합니다. input은 저희 경우 [1, 64, 4]입니다 그래서 아마 지우 형꺼는 [1, 4 ,64라 permute 해야합니다.]

  onset_pred = (onset_outputs > 0.5).float()
  
  plot_pianoroll(x_reshape.permute(0,2,1), vel_outputs * onset_pred, out_file_pth = 'transcribed_sample_results/das.png') 
  '''
  결과값을 보여줍니다.x는 원래 인풋 out_file_pth를 통해 저장할 경로를 지정할 수 있습니다.
  이때 정답값 vel_outputs는 [batch, 9, 64] shape으로 나오고 저희는 한개만 넣을 거기 때문에 그대로 넣어주시면 됩니다.
  plot 함수에서 배치부분 squeeze 합니다.
  즉, vel_outputs / onset pred는 모델이 예측한 그대로 넣어주시면 됩니다. 
  
  그리고 x는 모델에 들어갈때 [batch, 64, 9]로 들어가지만 plot일때는 원래 x의 shape인 [batch, 4, 64] 형태로 넣어주시면 됩니다. 
  저희의 경우 [1, 4, 64] 형태로 plot에 넣어주면 됩니다!
  지금 모델에 들어가기 위해 batch 쪽을 unsqueeze 해줬어서, 
  plot 함수에서 squeeze해주기 떄문에 배치는 그대로 두셔도 됩니다. 즉 [batch, 4, 64] (본래 인풋 형태로 넣어주시면 됩니다.)
  
  '''
  midi = get_nine_midi(onset = onset_outputs, val = vel_outputs, bpm = bpm ) 
  #결과값을 midi로 저장합니다. 모델이 예측한걸 그대로 넣어주시면 됩니다. bpm의 경우 기본 120입니다.
  midi.write('transcribed_sample_results/test.mid')  
 