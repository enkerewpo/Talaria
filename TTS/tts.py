import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="../GLaDOS_01.wav", language="en")
# Text to speech to a file

print("device is set to: ", device)
print("start typing and each line will be converted to speech")

while True:
    text = input()
    if text == "exit":
        break
    tts.tts_to_file(text=text, speaker_wav="../GLaDOS_01.wav", language="en", file_path="output.wav")
    # load the wav and playit
    import sounddevice as sd
    import soundfile as sf
    data, fs = sf.read('output.wav')
    sd.play(data, fs)
    sd.wait()

# tts.tts_to_file(text="Hello world, I'm your ambient assistant!", speaker_wav="../GLaDOS_01.wav", language="en", file_path="output.wav")