from om_transliterator import Transliterator
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the transliterator
transliterator = Transliterator()

# Define input directory path for punctuation files
input_dir = "./OUTPUT/TEXT/02_Punctuation"

# List all .txt files ending with punctuation.txt in the directory
punctuation_files = [f for f in os.listdir(input_dir) if f.endswith("punctuation.txt")]

# Define output directory for transliteration files
output_dir = "./OUTPUT/TEXT/03_Transliteration"
os.makedirs(output_dir, exist_ok=True)

def process_punctuation_file(punctuation_file):
    input_file_path = os.path.join(input_dir, punctuation_file)
    output_file_path = os.path.join(output_dir, punctuation_file.replace("punctuation.txt", "transliteration.txt"))
    
    # Read the original text from the file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the line with "Outputs:" and read the subsequent lines
    start_reading = True
    original_text = ""
    for line in lines:
        if start_reading:
            original_text += line.strip() + " "
        # if "Outputs:" in line:
        #     start_reading = True

    # Perform transliteration
    transliterated_text = transliterator.knda_to_latn(original_text.strip())

    # Write the transliterated text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(transliterated_text)

    print(f"Transliteration results saved to: {output_file_path}")

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_punctuation_file, punctuation_file) for punctuation_file in punctuation_files]

    for future in as_completed(futures):
        future.result()  # Retrieve the result to catch any exceptions

print("All transliterations are done.")
