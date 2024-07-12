import os
import subprocess
import glob
import torch
import torchaudio
from IPython import display as disp
from torchaudio.transforms import Resample
from denoiser.dsp import convert_audio
from denoiser import pretrained
import soundfile
from pystoi import stoi

# Installations
######################################################################
subprocess.run(["pip3", "install", "IPython"])
subprocess.run(["pip3", "install", "glob2"])
subprocess.run(["pip3", "install", "soundfile"])
subprocess.run(["pip3", "install", "torchaudio", "--upgrade"])
subprocess.run(["pip3", "install", "transformers", "--upgrade"])
subprocess.run(["pip3", "install", "pystoi"])
######################################################################

def dns_64(input_file, output_dir):
    LIST_OF_AUDIO_FILES = glob.glob(input_file)
    model = pretrained.dns64()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move the model to the appropriate device
    model = model.to(device)

    for audio_file in LIST_OF_AUDIO_FILES:
        wav, sr = torchaudio.load(audio_file)
        wav = convert_audio(wav, sr, model.sample_rate, model.chin)
        
        # Move the input tensor to the appropriate device
        wav = wav.to(device)

        with torch.no_grad():
            denoised = model(wav[None])[0]
            
            # Move the output back to CPU for display and saving
            denoised_cpu = denoised.cpu()
            
            disp.display(disp.Audio(denoised_cpu.numpy(), rate=model.sample_rate))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "1.mp3")
            torchaudio.save(output_file, denoised_cpu, model.sample_rate)

            # Compute STOI score
            denoised_np = denoised_cpu.numpy()[0]
            wav_np = wav.cpu().numpy()[0]
            stoi_score = stoi(wav_np, denoised_np, model.sample_rate, extended=False)
            print(f"STOI Score: {stoi_score}")

if __name__ == "__main__":
    input_file = r"./INPUT/AUDIO/audio.mp3"  # Using forward slashes for compatibility
    output_dir = "./INPUT_MP3S"  # Specify the output directory

    dns_64(input_file, output_dir)