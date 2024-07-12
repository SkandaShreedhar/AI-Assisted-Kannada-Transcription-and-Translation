import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")
model = Wav2Vec2ForCTC.from_pretrained("amoghsgopadi/wav2vec2-large-xlsr-kn")

# Load and preprocess your audio file
def preprocess_audio(file_path):
    speech_array, sampling_rate = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

# Function to transcribe audio and save the output to a text file
def transcribe_and_save(audio_path, output_path):
    speech = preprocess_audio(audio_path)
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription[0])
    print(f"Transcription saved to: {output_path}")

# Paths to your directories
audio_dir = "./OUTPUT/AUDIO/1.mp3_CUTS"
output_dir = "./OUTPUT/TEXT/01a_Transcription"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .mp3 files in the directory
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

# Process each audio file
for i, audio_file in enumerate(audio_files):
    audio_path = os.path.join(audio_dir, audio_file)
    output_file = os.path.join(output_dir, f"{i+1:02d}a_transcription.txt")
    
    try:
        transcribe_and_save(audio_path, output_file)
    except OSError as e:
        print(f"Error loading audio file {audio_file}: {e}")
    except Exception as e:
        print(f"Error during inference for {audio_file}: {e}")
