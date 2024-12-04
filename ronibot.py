import os
from gpt4all import GPT4All
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
import pyttsx3
import numpy as np

# Ruta al modelo GPT4All
model_path = "d:/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"

# Verificar si el modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo del modelo no existe en la ruta especificada: {model_path}")

# Cargar el modelo GPT4All
print(f"Cargando el modelo GPT4All desde: {model_path}")
try:
    gpt = GPT4All(model_path)
except Exception as e:
    print(f"Error al cargar el modelo GPT4All: {e}")
    exit(1)

# Ruta al modelo VOSK
vosk_model_path = "D:/gpt4all/vosk/vosk-model-small-es-0.42/vosk-model-small-es-0.42"
vosk_model = Model(vosk_model_path)

# Inicializar síntesis de voz
engine = pyttsx3.init()

# Contexto de Roni
roni_context = """
Hola, soy Roni, tu amigo virtual. Respuesta breve y simple. Estoy aquí para ayudarte, enseñarte y acompañarte siempre. ¡Me encanta jugar contigo y ayudarte a ser feliz! Siempre responderé en español, de manera amable y clara. Recuerda, mi trabajo es hacer que aprendas mientras te diviertes.
"""

# Función para hablar fragmentos de texto
def speak_fragment(fragment):
    print(f"Roni (fragmento): {fragment}")
    engine.say(fragment)
    engine.runAndWait()

# Función para procesar el audio y devolver texto transcrito
def process_audio():
    print("¡Escuchando entrada de voz!")
    sample_rate = 16000
    buffer_duration = 1  # Grabar en fragmentos de 1 segundo
    max_silence_duration = 3  # Detener grabación tras 3 segundos de silencio
    audio_data = []  # Lista para almacenar fragmentos grabados
    silence_counter = 0  # Contador para medir silencio continuo

    recognizer = KaldiRecognizer(vosk_model, sample_rate)

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

# Función para generar y hablar en streaming
def generate_and_speak_streaming(input_text):
    print("Generando respuesta en tiempo real...")
    try:
        for fragment in gpt.generate(input_text, streaming=True):  # Habilitar streaming
            speak_fragment(fragment)
    except Exception as e:
        print(f"Error en el streaming de respuesta: {e}")
        speak("Hubo un error al generar la respuesta. Intenta nuevamente.")

# Bucle principal para escuchar y responder continuamente
try:
    print("\n=== Roni: Tu Amigo Virtual ===")
    print("Escuchando activamente para procesar todo lo que digas.\n")
    while True:
        transcribed_text = process_audio()
        if not transcribed_text.strip():
            print("No se detectó texto relevante. Intentando nuevamente...")
            continue

        # Combinar el contexto con la transcripción para GPT4All
        combined_input = f"{roni_context}\n\nPregunta: {transcribed_text}"
        print(f"Tú: {transcribed_text}")

        # Generar y hablar la respuesta en tiempo real
        generate_and_speak_streaming(combined_input)
except KeyboardInterrupt:
    print("\nInterrumpido por el usuario. Cerrando Roni.")
    speak("Adiós, gracias por hablar conmigo. Hasta la próxima.")
