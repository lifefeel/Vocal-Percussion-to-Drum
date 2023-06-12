import argparse
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import torchaudio
from timeit import default_timer as timer
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
onset_model.load_state_dict(torch.load('onset_model_noQZ.pt'))
velocity_model.load_state_dict(torch.load('velocity_model_noQZ.pt'))

spec_converter = SpecConverter(sr=44100, n_fft=512, hop_length=128, n_mels=128)

def predict(audio_path):
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
    return dense_image, onset_image, pred_time

parser = argparse.ArgumentParser()
parser.add_argument('--share', type=str2bool, default='False')
args = parser.parse_args()

demo = gr.Interface(fn=predict,
                    inputs=gr.Audio(type="filepath"),
                    outputs=[gr.Image(label="densed_drumroll"),
                             gr.Image(label="onset_pred_cleaned"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    cache_examples=False
                    )

demo.launch(debug=False, share=args.share)