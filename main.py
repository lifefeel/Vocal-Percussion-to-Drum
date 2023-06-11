import gradio as gr
import torch
import torchaudio
from timeit import default_timer as timer
from data_setups import audio_preprocess, resample

device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100
AUDIO_LEN = 2.90
model = torch.load("torch_efficientnet_fold2_CNN.pth", map_location=torch.device('cpu'))
LABELS = [
    "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar", "Organ", "Piano", "Saxophone", "Trumpet", "Violin", "Voice"
]
example_list = [
    "audio_samples/pmta.wav", 
    "audio_samples/pmtati.wav", 
    "audio_samples/tita.wav", 
    "audio_samples/titi.wav"
]


def predict(audio_path):
    start_time = timer()
    wavform, sample_rate = torchaudio.load(audio_path)
    wav = resample(wavform, sample_rate, SAMPLE_RATE)
    if len(wav) > int(AUDIO_LEN * SAMPLE_RATE):
        wav = wav[:int(AUDIO_LEN * SAMPLE_RATE)]
    else:
        print(f"input length {len(wav)} too small!, need over {int(AUDIO_LEN * SAMPLE_RATE)}")
        return
    img = audio_preprocess(wav, SAMPLE_RATE).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    pred_labels_and_probs = {LABELS[i]: float(pred_probs[0][i]) for i in range(len(LABELS))}
    pred_time = round(timer() - start_time, 5)
    dense_image = "transcribed_sample_results/densed_drumroll_pmtati.png"
    onset_image = "transcribed_sample_results/onset_pred_cleaned_pmtati.png"
    # return pred_labels_and_probs, pred_time
    return dense_image, onset_image, pred_time

demo = gr.Interface(fn=predict,
                    inputs=gr.Audio(type="filepath"),
                    outputs=[gr.Image(label="densed_drumroll"),
                             gr.Image(label="onset_pred_cleaned"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    cache_examples=False
                    )

demo.launch(debug=False)