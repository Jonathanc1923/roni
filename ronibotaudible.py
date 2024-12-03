from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
import pyttsx3
import numpy as np

# Ruta al modelo VOSK
model_path = "D:/gpt4all/vosk/vosk-model-small-es-0.42/vosk-model-small-es-0.42"
model = Model(model_path)

# Inicializar síntesis de voz
engine = pyttsx3.init()

# Lista de variaciones de "Roni"
variations = [
    "roni", "ronnie", "ronny", "ronié", "roñi", "rónny", "rhonnie", "roonie",
    "ronie", "rónie", "rónni", "roñie", "rhonny", "rhoni", "rowni", "roné",
    "ruoni", "ronai", "rhonai", "ronéy", "rónai", "roní", "ronié", "rawnie",
    "rony", "ronié", "rowny", "ronnie"
]

# Función para hablar
def speak(text):
    print(f"Roni (hablando): {text}")
    engine.say(text)
    engine.runAndWait()

# Escuchar continuamente para detectar la palabra clave
def listen_for_keyword():
    print("Diga algo (por ejemplo, 'Roni') para activar:")
    recognizer = KaldiRecognizer(model, 16000)  # Configurar reconocedor
    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16") as stream:
        while True:
            data, _ = stream.read(4000)  # Leer fragmentos de audio
            data_array = np.frombuffer(data, dtype="int16")  # Convertir a array de Numpy
            data_bytes = data_array.tobytes()  # Convertir Numpy a bytes
            if recognizer.AcceptWaveform(data_bytes):  # Enviar a VOSK
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print(f"Texto detectado: {text}")
                if any(variant in text.lower() for variant in variations):  # Detectar cualquier variación
                    print("¡Palabra clave 'Roni' detectada!")
                    process_audio()
                    break

def process_audio():
    print("¡Escuchando entrada de voz!")
    sample_rate = 16000
    buffer_duration = 1  # Grabar en fragmentos de 1 segundo
    max_silence_duration = 3  # Detener grabación tras 3 segundos de silencio
    audio_data = []  # Lista para almacenar fragmentos grabados
    silence_counter = 0  # Contador para medir silencio continuo

    recognizer = KaldiRecognizer(model, sample_rate)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        print("Hable ahora. La grabación se detendrá cuando haya silencio.")
        while True:
            data, _ = stream.read(int(sample_rate * buffer_duration))
            audio_data.append(data)

            # Detectar si el fragmento actual está en silencio
            if np.max(np.abs(data)) < 1000:  # Ajustar umbral según tu entorno
                silence_counter += buffer_duration
            else:
                silence_counter = 0  # Restablecer si se detecta audio

            # Detener grabación si se supera la duración máxima de silencio
            if silence_counter >= max_silence_duration:
                print("Grabación detenida por silencio.")
                break

    # Procesar los datos acumulados con VOSK
    transcribed_text = []
    for chunk in audio_data:
        if recognizer.AcceptWaveform(chunk.tobytes()):
            result = json.loads(recognizer.Result())
            transcribed_text.append(result.get("text", ""))

    final_text = " ".join(transcribed_text).strip()
    print(f"Transcripción final: {final_text}")
    return final_text

# Ejecutar el sistema
try:
    listen_for_keyword()
except KeyboardInterrupt:
    print("Interrumpido por el usuario. Cerrando Roni.")
