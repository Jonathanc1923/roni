import os
from gpt4all import GPT4All

# Ruta completa al modelo descargado
model_path = "C:/Users/Album Magico/AppData/Local/nomic.ai/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"
## model_path = "D:/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# Verificar si el archivo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo del modelo no existe en la ruta especificada: {model_path}")

# Inicializar GPT4All con el modelo
print(f"Cargando el modelo GPT4All desde: {model_path}")
gpt = GPT4All(model_path)

# Configuración inicial del idioma
gpt.generate("Por favor, responde siempre en español. No mezcles otros idiomas.")

# Bucle interactivo para chat
print("\n=== GPT4All Conversación ===")
print("Escribe 'salir' para terminar.\n")

while True:
    try:
        user_input = input("Tú: ")
        if user_input.lower() == "salir":
            print("¡Adiós!")
            break

        # Generar respuesta
        response = gpt.generate(user_input)
        print(f"GPT4All: {response}\n")

    except Exception as e:
        print(f"Error al generar respuesta: {e}")
        break
