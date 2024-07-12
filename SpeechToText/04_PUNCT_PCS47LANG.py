from punctuators.models import PunctCapSegModelONNX
from typing import List
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Instantiate the model
m = PunctCapSegModelONNX.from_pretrained("pcs_47lang")

# Define input directory path
input_dir = "./OUTPUT/TEXT/01a_Transcription"

# List all .csv files ending with transcription.csv in the directory
transcription_files = [f for f in os.listdir(input_dir) if f.endswith("transcription.txt")]

# Define output directory for punctuation files
output_dir = "./OUTPUT/TEXT/02_Punctuation"
os.makedirs(output_dir, exist_ok=True)

def process_transcription_file(transcription_file):
    input_file = os.path.join(input_dir, transcription_file)
    
    # Read input texts from the file
    input_texts: List[str] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            input_texts.append(line.strip())  # Remove leading/trailing whitespaces

    # Run inference
    results: List[List[str]] = m.infer(input_texts)

    # Generate output file name
    output_file_name = transcription_file.replace("transcription.txt", "punctuation.txt")
    output_file = os.path.join(output_dir, output_file_name)

    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        for output_texts in results:
            for text in output_texts:
                f.write(f"{text}\n")
            f.write("\n")

    print(f"Results saved to: {output_file}")

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_transcription_file, transcription_file) for transcription_file in transcription_files]

    for future in as_completed(futures):
        future.result()  # Retrieve the result to catch any exceptions

print("All punctuations are done.")
