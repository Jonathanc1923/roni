from flask import Flask, request, jsonify, render_template, send_file
from gpt4all import GPT4All
import whisper
import pyttsx3
import wave
import os
import json
import threading
import time
import numpy as np
import subprocess

app = Flask(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Ruta relativa al modelo GPT
model_path = os.path.join(BASE_DIR, "modelos", "Llama-3.2-3B-Instruct-Q4_0.gguf")

# Ruta relativa a FFmpeg
ffmpeg_path = os.path.join(BASE_DIR, "ffmpeg", 
                           "ffmpeg-2024-11-28-git-bc991ca048-full_build", 
                           "ffmpeg-2024-11-28-git-bc991ca048-full_build", 
                           "bin", "ffmpeg.exe")

# Ruta relativa al modelo Whisper
whisper_model_path = os.path.join(BASE_DIR, "whisper", "small.pt")

# Imprimir rutas para verificar
print("Model Path:", model_path)
print("FFmpeg Path:", ffmpeg_path)
print("Whisper Model Path:", whisper_model_path)


# Configurar la variable de entorno para FFmpeg
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]

# Inicialización de modelos
print(f"Cargando modelo GPT4All desde: {model_path}")
gpt = GPT4All(model_path)
print("Cargando modelo Whisper...")
whisper_model = whisper.load_model("small", download_root=os.path.dirname(whisper_model_path))
engine = pyttsx3.init()

# Archivos temporales
TEMP_AUDIO_WAV = os.path.join(BASE_DIR, "temp_audio.wav")
TEMP_AUDIO_CONVERTED = TEMP_AUDIO_WAV.replace(".wav", "_converted.wav")
TEMP_AUDIO_RESPONSE = os.path.join(BASE_DIR, "response.wav")

# Configuración de limpieza de archivos
RESPONSE_LIFETIME = 60  # Tiempo en segundos antes de eliminar el audio de respuesta
last_access_time = {}

def cleanup_temp_files():
    """Limpia archivos temporales después de un tiempo definido."""
    while True:
        current_time = time.time()
        for file_path, last_access in list(last_access_time.items()):
            if current_time - last_access > RESPONSE_LIFETIME:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Archivo eliminado: {file_path}")
                del last_access_time[file_path]
        time.sleep(10)

# Iniciar la limpieza en un hilo separado
threading.Thread(target=cleanup_temp_files, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process-audio", methods=["POST"])
def process_audio():
    try:
        # Verificar si se envió un archivo
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        # Guardar el archivo recibido
        audio_file = request.files["file"]
        audio_file.save(TEMP_AUDIO_WAV)

        # Validar si el archivo tiene datos de audio
        with wave.open(TEMP_AUDIO_WAV, "rb") as wav_file:
            params = wav_file.getparams()
            duration = params.nframes / params.framerate
            print(f"Duración del audio: {duration:.2f} segundos")

            if duration < 0.5:
                return jsonify({"error": "El audio es demasiado corto para transcribir"}), 400

            if params.framerate != 16000 or params.nchannels != 1:
                # Convertir el audio a formato mono y 16 kHz usando ffmpeg
                ffmpeg_command = [
                    ffmpeg_path,
                    "-i", TEMP_AUDIO_WAV,
                    "-ac", "1",
                    "-ar", "16000",
                    TEMP_AUDIO_CONVERTED
                ]
                print(f"Ejecutando FFmpeg: {' '.join(ffmpeg_command)}")
                subprocess.run(ffmpeg_command, check=True)
                print(f"Audio convertido a formato mono y 16 kHz: {TEMP_AUDIO_CONVERTED}")
            else:
                TEMP_AUDIO_CONVERTED = TEMP_AUDIO_WAV  # Ya está en el formato correcto

        # Transcribir audio con Whisper
        print("Iniciando transcripción con Whisper...")
        transcription_result = whisper_model.transcribe(TEMP_AUDIO_CONVERTED, fp16=False)
        transcription = transcription_result.get("text", "").strip()

        if not transcription:
            return jsonify({"error": "No se pudo transcribir el audio"}), 400

        print(f"Transcripción completada: {transcription}")

        # Generar respuesta con GPT4All
        response = "".join(gpt.generate(transcription, streaming=False))

        # Generar archivo de respuesta de audio
        engine.save_to_file(response, TEMP_AUDIO_RESPONSE)
        engine.runAndWait()

        # Registrar el tiempo de último acceso al archivo
        last_access_time[TEMP_AUDIO_RESPONSE] = time.time()

        return jsonify({"transcription": transcription, "response": response})

    except subprocess.CalledProcessError as ffmpeg_error:
        print(f"Error al ejecutar FFmpeg: {ffmpeg_error}")
        return jsonify({"error": "Error al procesar el archivo de audio"}), 500
    except Exception as e:
        print(f"Error inesperado: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/audio-response", methods=["GET"])
def audio_response():
    if os.path.exists(TEMP_AUDIO_RESPONSE):
        # Actualizar tiempo de último acceso
        last_access_time[TEMP_AUDIO_RESPONSE] = time.time()
        return send_file(TEMP_AUDIO_RESPONSE, as_attachment=False, mimetype="audio/wav")
    return jsonify({"error": "El audio de respuesta no está disponible"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
