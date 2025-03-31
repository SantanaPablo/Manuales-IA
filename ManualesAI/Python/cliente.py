# cliente.py
import requests
import json

while True:
    pregunta = input("Pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
        break

    url = f"http://127.0.0.1:8000/buscar/?pregunta={pregunta}"
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        print("\nRespuesta: ", end="", flush=True)
        for line in response.iter_lines():
            if line:  # Evita líneas vacías
                try:
                    # Se intenta cargar la línea como JSON y obtener el valor de "respuesta"
                    data = json.loads(line.decode("utf-8"))
                    print(data.get("respuesta", "No se encontró respuesta."), end="", flush=True)
                except json.JSONDecodeError:
                    # En caso de error al decodificar JSON, se ignora la línea
                    continue
        print("\n")  # Salto de línea al terminar
    else:
        print("\nError en la consulta.")
