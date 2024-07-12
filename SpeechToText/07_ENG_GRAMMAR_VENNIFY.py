from transformers import pipeline
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def grammar_check(text):
    # Initialize the text2text-generation pipeline with the grammar correction model
    corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
    
    # Generate the corrected text
    corrected = corrector(text, max_length=len(text) + 50)[0]['generated_text']
    
    return corrected

def get_text_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    return content

def write_text_to_file(filepath, text):
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(text)

# Define input and output directories
input_dir = "./OUTPUT/TEXT/04_Translation"
output_dir = "./OUTPUT/TEXT/05_English_Grammar"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# List all .txt files ending with translation.txt in the input directory
translation_files = [f for f in os.listdir(input_dir) if f.endswith("translation.txt")]

def process_translation_file(translation_file):
    file_path = os.path.join(input_dir, translation_file)
    output_path = os.path.join(output_dir, translation_file.replace("translation.txt", "english_grammar.txt"))

    try:
        # Get text from the translation file
        input_text = get_text_from_file(file_path)

        # Perform grammar check
        corrected_text = grammar_check(input_text)

        # Write corrected text to the output file
        write_text_to_file(output_path, corrected_text)

        print(f"Original text from {translation_file}: {input_text}")
        print(f"Corrected text saved to {output_path}: {corrected_text}")
    except Exception as e:
        print(f"Error processing {translation_file}: {e}")

# Use ThreadPoolExecutor to process files in parallel
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_translation_file, translation_file) for translation_file in translation_files]

    for future in as_completed(futures):
        future.result()  # Retrieve the result to catch any exceptions

print("All grammar corrections are done.")
