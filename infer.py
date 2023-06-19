from model import drum_gen_model
from infer_util import plot_pianoroll, get_nine_midi
import torch 
import argparse
import random

# 현재 input으로 들어와야할 지우형의 input이 파일 형태로올지 모르겠어서 남겨놨ㅅ브니다.


def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bpm', type=int, default='120')
  parser.add_argument('--style', type=str, default='rock')
  
  return parser



if __name__ == "__main__":
  args = get_argument_parser().parse_args()

  bpm = args.bpm
  style = args.style
  print(style)
  
  style2idx = {'gospel': 0, 'blues': 1,'afrobeat': 2,'jazz': 3,'country': 4,'funk': 5,'rock': 6,
                      'hiphop': 7,'latin': 8,'dance': 9,'middleeastern': 10,'reggae': 11,'neworleans': 12,
                      'punk': 13,'pop': 14,'soul': 15,'highlife': 16,'afrocuban': 17}

  
  style_idx = torch.tensor([style2idx[style]], dtype=torch.long)
  model = drum_gen_model()
  model.load_state_dict(torch.load('./models/25model_100.pt')) # 아직 가중치가 없어서 추가할 예정입니다. 모델구조와 가중치는 따로 수정할 수 있도록 해주시면 감사하겠습니다! 

  model.eval()
  with torch.no_grad():
    x_reshape = torch.randn(1, 64, 4)
    vel_outputs, onset_outputs = model(x = x_reshape, style_idx= style_idx, teacher_forcing_ratio = 0.0) 
    # input이 들어와야합니다. input은 저희 경우 [1, 64, 4]입니다 그래서 아마 지우 형꺼는 [1, 4 ,64라 permute 해야합니다.]

  onset_pred = (onset_outputs > 0.5).float()
  
  vel_outputs = torch.where(vel_outputs > 60/127, vel_outputs, torch.zeros_like(vel_outputs))
  vel_onset = torch.where(vel_outputs > 60/127, torch.ones_like(vel_outputs), torch.zeros_like(vel_outputs))

  plot_pianoroll(x_reshape, vel_outputs, out_file_pth = '/home/daewoong/userdata/drum_transcription/inference/dasds.png') 
  #plot_pianoroll(pmta, vel_outputs * onset_pred, out_file_pth = '/home/daewoong/userdata/drum_transcription/inference/dass.png') 
  '''
  결과값을 보여줍니다.x는 원래 인풋 out_file_pth를 통해 저장할 경로를 지정할 수 있습니다.
  이때 정답값 vel_outputs는 [batch, 9, 64] shape으로 나오고 저희는 한개만 넣을 거기 때문에 그대로 넣어주시면 됩니다.
  plot 함수에서 배치부분 squeeze 합니다.
  즉, vel_outputs / onset pred는 모델이 예측한 그대로 넣어주시면 됩니다. 
  
  그리고 x는 모델에 들어갈때 [1, 64, 9]로 들어가지만 plot일때는 원래 x의 shape인 [1, 4, 64] 형태로 넣어주시면 됩니다. 
  저희의 경우 [1, 4, 64] 형태로 plot에 넣어주면 됩니다!
  지금 모델에 들어가기 위해 batch 쪽을 unsqueeze 해줬어서, 
  plot 함수에서 squeeze해주기 떄문에 배치는 그대로 두셔도 됩니다. 즉 [1, 4, 64] (본래 인풋 형태로 넣어주시면 됩니다.)
  
  '''
  midi = get_nine_midi(onset = vel_onset, val = vel_outputs, bpm = bpm ) 
  #저장할 경로를 지정하면됩니다!
 