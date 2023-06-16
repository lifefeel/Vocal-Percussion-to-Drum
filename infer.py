from model import drum_generation_model
from infer_util import plot_pianoroll, get_nine_midi
import torch 
import argparse


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
  
  style2idx = {'gospel': 0, 'blues': 1,'afrobeat': 2,'jazz': 3,'country': 4,'funk': 5,'rock': 6,
                      'hiphop': 7,'latin': 8,'dance': 9,'middleeastern': 10,'reggae': 11,'neworleans': 12,
                      'punk': 13,'pop': 14,'soul': 15,'highlife': 16,'afrocuban': 17}

  
  style_idx = torch.tensor(style2idx[style], dtype=torch.long)

  model = drum_generation_model()
  model.load_state_dict(torch.load('./models/17model_100.pt')) # 아직 가중치가 없어서 추가할 예정입니다. 모델구조와 가중치는 따로 수정할 수 있도록 해주시면 감사하겠습니다! 
  
  model.eval()
  with torch.no_grad():
    x_reshape = x.permute(0)
    vel_outputs, onset_outputs = model(x = x_reshape, style_idx= style_idx, teacher_forcing_ratio = 0.0) 
    # input이 들어와야합니다. input은 저희 경우 [1, 64, 4]입니다 그래서 아마 지우 형꺼는 [1, 4 ,64라 permute 해야합니다.]
    
  plot_pianoroll(x, vel_outputs, out_file_pth) #결과값을 보여줍니다.x는 원래 인풋 out_file_pth를 통해 저장할 경로를 지정할 수 있습니다.
  
  midi = get_nine_midi(onset = onset_outputs, val = vel_outputs, bpm = bpm ) #결과값을 midi로 저장합니다.
  midi.write('./OOT.mid')   #저장할 경로를 지정하면됩니다!
 