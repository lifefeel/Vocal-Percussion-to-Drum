import numpy as np 
import matplotlib.pyplot as plt
import pretty_midi

def plot_pianoroll(x, pred, out_file_pth=None):
  fig, ax = plt.subplots(3, 1, figsize=(15, 10))
  #dataset.paper_mapping_pitch, dataset.paper_idx2pitch, dataset.paper_mapping_inst, dataset.paper_idx2pitch_for_x
  ax[0].imshow(x)
  ax[0].set_ylabel('Input')
  # ax[0].set_yticklabels(['', 'Bass', '', 'Snare', '', 'Closed Hi-Hat', '', 'Open Hi-Hat'])
  ax[0].set_yticks(np.arange(4)) # add this line

  ax[0].set_yticklabels(['Bass','Snare', 'Closed Hi-Hat','Open Hi-Hat'])

  ax[1].imshow(pred)
  ax[1].set_ylabel('pred')
  ax[1].set_yticks(np.arange(9)) # add this line
  ax[1].set_yticklabels(['Bass','Snare','Closed Hi-Hat','High Floor Tom','Open Hi-Hat','Low-Mid Tom','Crash Cymbal','High Tom','Ride Cymbal'])

  plt.subplots_adjust(wspace=0.9, hspace=0.5)
  plt.save(out_file_pth)
  plt.close()
  
def get_nine_midi(onset, val, bpm, threshold = 0.5):
  pitch2idx = {0: 36, 1: 38, 2: 42, 3: 43, 4: 46, 5: 47, 6: 49, 7: 50, 8: 51}

  beats_per_measure = 8  # 분자: 하나의 마디에 포함된 비트 수
  beat_unit = 4          # 분모: 기본 비트 (4는 4분음표)

  # 비트당 시간 (초 단위) 계산
  beat_duration = 60 / bpm

  # 마디당 비트 수 계산
  measures = beats_per_measure / beat_unit if beat_unit != 0 else beats_per_measure

  # beat grid 생성
  beat_grid = []
  for measure in range(int(measures)):
      for beat in range(beats_per_measure):
          beat_time = (measure * beats_per_measure + beat) * beat_duration
          beat_grid.append(beat_time)
          
  beat_intervel = beat_grid[1] - beat_grid[0]
  beat_sixteen_intervel = beat_intervel / 4
  last_time = beat_grid[-1] + beat_intervel
  sixteen_beat_grid = np.arange(0, last_time, beat_sixteen_intervel)


  midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
  instrument = pretty_midi.Instrument(program=0, is_drum=True)
  
  onset = (onset >= threshold).float()
  
  for pitch in range(len(onset)):
    for idx, (onoff, velocity) in enumerate(zip(onset[pitch], val[pitch])):
      if onoff == 1:
        note = pretty_midi.Note(
            start = sixteen_beat_grid[idx],
            end = sixteen_beat_grid[idx+1] if idx+1 < len(sixteen_beat_grid) else sixteen_beat_grid[idx] + beat_sixteen_intervel,
            pitch = pitch2idx[pitch],
            velocity = int(velocity * 127) 
        )
        instrument.notes.append(note)
              
  midi.instruments.append(instrument)
  
  return midi
