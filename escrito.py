import os
from gpt4all import GPT4All

# Ruta al modelo
model_path = "d:/GPT4All/Llama-3.2-3B-Instruct-Q4_0.gguf"

# Verificar si el modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El archivo del modelo no existe en la ruta especificada: {model_path}")

# Cargar el modelo GPT4All
print(f"Cargando el modelo GPT4All desde: {model_path}")
gpt = GPT4All(model_path)

# Contexto incluido directamente en el código
roni_context = """
Hola, soy Roni, tu amigo virtual. Respuesta breve y simple. Estoy aquí para ayudarte, enseñarte y acompañarte siempre. ¡Me encanta jugar contigo y ayudarte a ser feliz! Siempre responderé en español, de manera amable y clara. Recuerda, mi trabajo es hacer que aprendas mientras te diviertes.

"""

print("\n=== Roni: Tu Amigo Virtual ===")
print("Escribe 'salir' para terminar.\n")

# Bucle interactivo para conversación
while True:
    user_input = input("Tú: ")
    if user_input.lower() == "salir":
        print("¡Adiós! Roni te dice que eres increíble, ¡hasta la próxima!")
        break

    # Combinar el contexto con la pregunta del usuario
    combined_input = f"{roni_context}\n\nPregunta: {user_input}"

    # Generar la respuesta
    response = gpt.generate(combined_input)
    print(f"Roni: {response.strip()}\n")