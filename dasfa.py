from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
import numpy as np
import os

# Ruta al modelo
model_path = os.path.abspath("D:/gpt4all/vosk/vosk-model-small-es-0.42/vosk-model-small-es-0.42")
model = Model(model_path)

# Crear el reconocedor
recognizer = KaldiRecognizer(model, 16000)

# Lista de variaciones de "Roni"
keyword_variations = ["roni", "ronnie", "ronny", "rony", "rooney", "rony"]

# Función para verificar si una palabra coincide con las variaciones
def is_keyword_detected(text):
    for word in keyword_variations:
        if word in text.lower():
            return True
    return False

# Función para escuchar y detectar palabras clave
def listen_for_keyword():
    print("Diga algo (por ejemplo, 'Roni') para probar:")
    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16") as stream:
        while True:
            # Leer datos de audio
            data, overflow = stream.read(4000)  # Captura 4000 frames de audio
            if overflow:
                print("Advertencia: desbordamiento de audio detectado.")
            
            # Convertir datos a numpy array y luego a bytes
            data_array = np.frombuffer(data, dtype="int16")  # Convertir buffer a numpy array
            data_bytes = data_array.tobytes()  # Convertir numpy array a formato bytes
            
            # Enviar datos al reconocedor
            if recognizer.AcceptWaveform(data_bytes):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print("Texto detectado:", text)
                
                # Verificar si se detecta una variación de la palabra clave
                if is_keyword_detected(text):
                    print("¡Palabra clave 'Roni' detectada!")
                    break

listen_for_keyword()
