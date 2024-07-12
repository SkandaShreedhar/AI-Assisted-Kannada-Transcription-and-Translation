from flask import Flask, jsonify, render_template, send_file, request, send_from_directory
from flask_cors import CORS
import subprocess
import os
import glob
import time
import csv
from pathlib import Path
from pydub import AudioSegment
import psutil
import traceback
app = Flask(__name__)
CORS(app)

OUTPUT_TEXT_FOLDER = './OUTPUT/TEXT'
OUTPUT_AUDIO_FOLDER = './OUTPUT/AUDIO'
INPUT_AUDIO_FOLDER = './INPUT/AUDIO'
RESULTS_FOLDER = Path('./RESULTS')

models1 = [
    '01_DENOISER_DNS_64.py',
    '02_SPEECH_SEG_FFMPEG.py',
    '03a_ASR_ADDY88_W2V.py',
    '04_PUNCT_PCS47LANG.py',
    '05_TRANSLIT_OM.py',
    '06_TRANSLATE_AI4BHARAT.py',
    '07_ENG_GRAMMAR_VENNIFY.py'
]

models2 = [
    '01_DENOISER_DNS_64.py',
    '02_SPEECH_SEG_FFMPEG.py',
    '03b_ASR_VAS22_WHISPER.py',
    '04_PUNCT_PCS47LANG.py',
    '05_TRANSLIT_OM.py',
    '06_TRANSLATE_AI4BHARAT.py',
    '07_ENG_GRAMMAR_VENNIFY.py'
]

models3 = [
    '01_DENOISER_DNS_64.py',
    '02_SPEECH_SEG_FFMPEG.py',
    '03c_ASR_AMO_WAV2VEC.py',
    '04_PUNCT_PCS47LANG.py',
    '05_TRANSLIT_OM.py',
    '06_TRANSLATE_AI4BHARAT.py',
    '07_ENG_GRAMMAR_VENNIFY.py'
]

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/run-script/<int:model_index>/<int:script_index>', methods=['POST'])
def run_script(model_index, script_index):
    model_selection = {1: models1, 2: models2, 3: models3}
    models = model_selection[int(model_index)]
    if script_index < 0 or script_index >= len(models):
        return jsonify({"error": "Invalid script index"}), 400
    model = models[script_index]
    info_file = 'info.txt'
    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        subprocess.run(['python', model], check=True)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Collect memory and CPU usage
        memory_info = process.memory_info()
        cpu_usage = process.cpu_percent(interval=1)

        with open(info_file, 'a') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
            f.write(f"Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB\n")
            f.write(f"CPU Usage: {cpu_usage:.2f}%\n\n")

        time.sleep(2)  # Give some time for file writing to complete
        return jsonify(get_output(script_index))
    except subprocess.CalledProcessError as e:
        error_message = f"Error running {model}: {str(e)}"
        app.logger.error(error_message)
        app.logger.error(traceback.format_exc())
        return jsonify({"error": error_message}), 500
    except Exception as e:
        error_message = f"Unexpected error occurred: {str(e)}"
        app.logger.error(error_message)
        app.logger.error(traceback.format_exc())
        return jsonify({"error": error_message}), 500


def get_output(script_index):
    if script_index == 0:
        return {"speech_separation": read_audio_files(os.path.join(OUTPUT_AUDIO_FOLDER, '1.mp3_CUTS', '*.mp3'))}
    elif script_index == 1:
        return {"transcription": read_files(os.path.join(OUTPUT_TEXT_FOLDER, '01a_Transcription', '*.txt'))}
    elif script_index == 2:
        return {"punctuation": read_files(os.path.join(OUTPUT_TEXT_FOLDER, '02_Punctuation', '*.txt'))}
    elif script_index == 3:
        return {"transliteration": read_files(os.path.join(OUTPUT_TEXT_FOLDER, '03_Transliteration', '*.txt'))}
    elif script_index == 4:
        return {"translation": read_files(os.path.join(OUTPUT_TEXT_FOLDER, '04_Translation', '*.txt'))}
    elif script_index == 5:
        return {"grammar": read_files(os.path.join(OUTPUT_TEXT_FOLDER, '05_English_Grammar', '*.txt'))}
    else:
        return {}

def read_files(filepath_pattern):
    results = {}
    for filepath in glob.glob(filepath_pattern):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                results[filename] = file.read()
        except FileNotFoundError:
            results[filename] = "File not found"
    return results

def read_audio_files(filepath_pattern):
    results = {}
    for filepath in glob.glob(filepath_pattern):
        filename = os.path.basename(filepath)
        results[filename] = f'/audio/{filename}'
    return results

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(os.path.join(OUTPUT_AUDIO_FOLDER, '1.mp3_CUTS', filename), mimetype='audio/mp3')

@app.route('/save-content', methods=['POST'])
def save_content():
    data = request.json
    file_id = data['id']
    content = data['content']
    
    # Parse the file_id to get the folder and filename
    parts = file_id.split('_')
    folder = parts[0]
    filename = '_'.join(parts[1:])
    
    filepath = os.path.join(OUTPUT_TEXT_FOLDER, folder, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        return jsonify({"message": "Content saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error saving content: {str(e)}"}), 500

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    good_transcriptions = data[0].get('goodTranscriptions', [])
    bad_transcriptions = data[0].get('badTranscriptions', [])

    try:
        save_transcriptions(good_transcriptions, 'Good_Transcriptions.txt')
        save_transcriptions(bad_transcriptions, 'Bad_Transcriptions.txt')
        return jsonify({"message": "Submissions saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Error saving submissions: {str(e)}"}), 500

def save_transcriptions(transcriptions, filename):
    filepath = RESULTS_FOLDER / filename
    RESULTS_FOLDER.mkdir(exist_ok=True)
    
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Filename', 'Content'])
        for transcription in transcriptions:
            writer.writerow([transcription['filename'], transcription['content']])

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if audio_file:
        # Always save as audio.mp3 in the INPUT/AUDIO folder
        temp_path = os.path.join(INPUT_AUDIO_FOLDER, 'temp_audio')
        filename = 'audio.mp3'
        upload_path = os.path.join(INPUT_AUDIO_FOLDER, filename)
        
        # Ensure the INPUT_AUDIO_FOLDER exists
        os.makedirs(INPUT_AUDIO_FOLDER, exist_ok=True)
        
        # Save the temporary file
        audio_file.save(temp_path)
        
        # Convert the temporary audio file to a standard MP3 format
        audio = AudioSegment.from_file(temp_path)
        audio.export(upload_path, format='mp3')
        
        # Remove the temporary file
        os.remove(temp_path)
        
        return jsonify({"success": True, "message": "Audio file saved successfully as audio.mp3"}), 200

if __name__ == '__main__':
    app.run(debug=True)
