import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

model_id = "addy88/wav2vec2-kannada-stt"

audio_dir = "./OUTPUT/AUDIO/1.mp3_CUTS"
# audio_dir = "./INPUT_MP3S"
output_dir = "./OUTPUT/TEXT/01a_Transcription"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .mp3 files in the directory
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

# Load the processor for inference
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

def process_audio_file(i, audio_file):
    audio_path = os.path.join(audio_dir, audio_file)
    
    try:
        audio, orig_freq = torchaudio.load(audio_path)

        # Ensure mono channel:
        audio = audio.mean(dim=0, keepdim=True)

        # Resample to model-compatible frequency (check documentation):
        resampler = torchaudio.transforms.Resample(orig_freq, 16_000)  # assuming 16 kHz
        audio = resampler(audio)

        # Prepare audio input for the model:
        audio_input = {"input_values": audio}

        inputs = processor(
            audio_input["input_values"].numpy(),
            return_tensors="pt",
            sampling_rate=16_000
        )

        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Remove <s> tokens from the transcription
        cleaned_transcription = transcription.replace("<s>", "").strip()

        # Save transcription to a file
        output_file = os.path.join(output_dir, f"{i+1:02d}a_transcription.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(cleaned_transcription)

        print(f"Transcription for {audio_file} saved to: {output_file}")

    except OSError as e:
        print(f"Error loading audio file {audio_file}: {e}")
    except Exception as e:
        print(f"Error during inference for {audio_file}: {e}")

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_audio_file, i, audio_file) for i, audio_file in enumerate(audio_files)]

    for future in as_completed(futures):
        future.result()  # Retrieve the result to catch any exceptions

print("All transcriptions are done.")
