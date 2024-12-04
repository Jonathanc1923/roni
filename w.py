import os
import whisper

# Ruta específica de FFmpeg
ffmpeg_path = r"D:\gpt4all\ffmpeg\ffmpeg-2024-11-28-git-bc991ca048-full_build\ffmpeg-2024-11-28-git-bc991ca048-full_build\bin\ffmpeg.exe"

# Configurar la variable de entorno para que Whisper use el FFmpeg local
os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]

# Cargar el modelo de Whisper desde la ubicación local
model_path = r"D:/gpt4all/whisper/small.pt"
print(f"Cargando modelo Whisper desde: {model_path}")
model = whisper.load_model(model_path)

# Ruta al archivo de audio (asegúrate de que esté en formato compatible)
audio_file = "temp_audio.wav"
print(f"Cargando audio: {audio_file}")

try:
    # Transcribir el audio
    print("Transcribiendo...")
    result = model.transcribe(audio_file, fp16=False)  # Usar CPU
    print("Texto transcrito:")
    print(result["text"])
except Exception as e:
    print(f"Error durante la transcripción: {e}")
