import argparse
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import torchaudio
from timeit import default_timer as timer
from infer_util import plot_pianoroll
from model import drum_generation_model
from model_zoo import RNNModel_onset, RNNModel_velocity
from transcription import Dense_onsets, SpecConverter, denoise, high_freq_content, plt_imshow, reducing_time_resolution

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
device = "cuda" if torch.cuda.is_available() else "cpu"

example_list = [
    "audio_samples/pmta.wav", 
    "audio_samples/pmtati.wav", 
    "audio_samples/tita.wav", 
    "audio_samples/titi.wav"
]

device = torch.device('cpu')
onset_model = RNNModel_onset(num_nmels=128, hidden_size=128).to(device)
velocity_model = RNNModel_velocity(num_nmels=128, hidden_size=128).to(device)

# load model
onset_model.load_state_dict(torch.load('models/onset_model_noQZ.pt'))
velocity_model.load_state_dict(torch.load('models/velocity_model_noQZ.pt'))

bpm = 120
style = 'rock'
style2idx = {'gospel': 0, 'blues': 1,'afrobeat': 2,'jazz': 3,'country': 4,'funk': 5,'rock': 6,
                      'hiphop': 7,'latin': 8,'dance': 9,'middleeastern': 10,'reggae': 11,'neworleans': 12,
                      'punk': 13,'pop': 14,'soul': 15,'highlife': 16,'afrocuban': 17}

model = drum_generation_model()
model.load_state_dict(torch.load('./models/17model_100.pt')) # 아직 가중치가 없어서 추가할 예정입니다. 모델구조와 가중치는 따로 수정할 수 있도록 해주시면 감사하겠습니다! 
model.eval()

spec_converter = SpecConverter(sr=44100, n_fft=512, hop_length=128, n_mels=128)

def transcribe(audio_path, transcription_output):
    start_time = timer()
    audio_path = Path(audio_path)
    sample_first, sr = torchaudio.load(audio_path)
    
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
    
    pred_time = round(timer() - start_time, 5)
    dense_image = f"transcribed_sample_results/densed_drumroll_{audio_path.stem}.png"
    onset_image = f"transcribed_sample_results/onset_pred_cleaned_{audio_path.stem}.png"
    # return pred_labels_and_probs, pred_time
    transcription_output = densed_drumroll[:,:64]
    
    return dense_image, onset_image, pred_time, transcription_output

def generate(x):
    style_idx = torch.tensor([style2idx[style]], dtype=torch.long)
    x = torch.from_numpy(x).unsqueeze(0)
    
    with torch.no_grad():
        x_reshape = x.permute(0, 2, 1)
        vel_outputs, onset_outputs = model(x = x_reshape, style_idx= style_idx, teacher_forcing_ratio = 0.0) 
        # input이 들어와야합니다. input은 저희 경우 [1, 64, 4]입니다 그래서 아마 지우 형꺼는 [1, 4 ,64라 permute 해야합니다.]

    onset_pred = (onset_outputs > 0.5).float()
    
    saved_image = 'transcribed_sample_results/das.png'
    plot_pianoroll(x_reshape.permute(0,2,1), vel_outputs * onset_pred, out_file_pth=saved_image) 
    
    return saved_image
    
parser = argparse.ArgumentParser()
parser.add_argument('--share', type=str2bool, default='False')
args = parser.parse_args()

with gr.Blocks() as demo:
    gr.Markdown("""# Vocal Percussion to Drum""")
    gr.Markdown("""## Transcription""")
    input_audio = gr.Audio(type="filepath")
    examples = gr.Examples(example_list, label="Examples", inputs=input_audio)
    run_button = gr.Button(value="Transcribe")
    
    gr.Markdown("""### Transcription Results""")
    with gr.Row():
        dense_image = gr.Image(label="densed_drumroll")
        onset_image = gr.Image(label="onset_pred_cleaned")
    
    time_label = gr.Number(label="Prediction time (s)")
    gr.Markdown("""## Generation""")
    generate_button = gr.Button(value="Generate")
    generated_image = gr.Image(label="generated_drumroll")
    
    transcription_output = gr.State()
    
    #
    # 버튼 클릭 이벤트
    #
    run_button.click(fn=transcribe, inputs=[input_audio, transcription_output], outputs=[dense_image, onset_image, time_label, transcription_output])
    generate_button.click(fn=generate, inputs=transcription_output, outputs=generated_image)

    demo.launch(debug=False, share=args.share)