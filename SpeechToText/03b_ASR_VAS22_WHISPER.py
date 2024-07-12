import torch
from transformers import pipeline
import os

# Define the directory containing the audio files
audio_dir = "./OUTPUT/AUDIO/1.mp3_CUTS"
output_dir = "./OUTPUT/TEXT/01a_Transcription"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .mp3 files in the directory
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]

# Set the device to CUDA if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the Whisper model for Kannada transcription
transcribe = pipeline(
    task="automatic-speech-recognition",
    model="vasista22/whisper-kannada-tiny",
    chunk_length_s=15,
    stride_length_s=5,
    batch_size=1,
    device=device
)

transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="kn", task="transcribe")

# Process each audio file
for i, audio_file in enumerate(audio_files):
    audio_path = os.path.join(audio_dir, audio_file)
    
    try:
        # Transcribe the audio file
        transcription = transcribe(audio_path)["text"]

        # Save transcription to a text file
        output_file = os.path.join(output_dir, f"{i+1:02d}a_transcription.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"Transcription for {audio_file} saved to: {output_file}")

    except OSError as e:
        print(f"Error loading audio file {audio_file}: {e}")
    except Exception as e:
        print(f"Error during inference for {audio_file}: {e}")
