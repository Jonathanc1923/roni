from flask import Flask, render_template
from flask_socketio import SocketIO
from gpt4all import GPT4All
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import pyttsx3
import wave
import json
import os

app = Flask(__name__)
socketio = SocketIO(app)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "modelos/Llama-3.2-3B-Instruct-Q4_0.gguf")
vosk_model_path = os.path.join(BASE_DIR, "vosk/vosk-model-small-es-0.42/vosk-model-small-es-0.42")
TEMP_AUDIO_RAW = os.path.join(BASE_DIR, "temp_input.raw")
TEMP_AUDIO_WAV = os.path.join(BASE_DIR, "temp_input.wav")
TEMP_AUDIO_RESPONSE = os.path.join(BASE_DIR, "response.wav")

# Inicialización de modelos
gpt = GPT4All(model_path)
vosk_model = Model(vosk_model_path)
engine = pyttsx3.init()

# Asegurar que pydub utiliza ffmpeg
AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Ajusta esta ruta según la instalación de ffmpeg

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("audio_data")
def handle_audio(audio_data):
    try:
        # Guardar audio recibido
        raw_audio_path = os.path.join(BASE_DIR, "debug_received_audio.webm")
        with open(raw_audio_path, "wb") as raw_file:
            raw_file.write(audio_data)

        # Convertir a formato WAV usando pydub
        try:
            audio_segment = AudioSegment.from_file(raw_audio_path, format="webm")
            audio_segment.export(TEMP_AUDIO_WAV, format="wav")
        except Exception as e:
            socketio.emit("error", f"Error al convertir a WAV: {str(e)}")
            return

        # Validar archivo WAV
        try:
            with wave.open(TEMP_AUDIO_WAV, "rb") as wav_file:
                if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000:
                    socketio.emit("error", "El archivo WAV no tiene el formato requerido (1 canal, 16 kHz).")
                    return
        except wave.Error as e:
            socketio.emit("error", f"Archivo WAV inválido: {str(e)}")
            return

        # Transcribir con VOSK
        recognizer = KaldiRecognizer(vosk_model, 16000)
        transcription = ""
        with wave.open(TEMP_AUDIO_WAV, "rb") as wav_file:
            while True:
                data = wav_file.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    transcription = result.get("text", "")

        if not transcription:
            socketio.emit("error", "No se pudo transcribir el audio.")
            return

        # Generar respuesta con GPT4All
        response = "".join(gpt.generate(transcription, streaming=False))

        # Convertir respuesta a audio
        engine.save_to_file(response, TEMP_AUDIO_RESPONSE)
        engine.runAndWait()

        # Enviar el audio de respuesta al cliente
        with open(TEMP_AUDIO_RESPONSE, "rb") as f:
            audio_response = f.read()
        socketio.emit("audio_response", audio_response)

    except Exception as e:
        socketio.emit("error", f"Error del servidor: {str(e)}")
    finally:
        # Limpieza de archivos temporales
        for temp_file in [raw_audio_path, TEMP_AUDIO_WAV, TEMP_AUDIO_RESPONSE]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    print("Servidor iniciado en http://127.0.0.1:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
