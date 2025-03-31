# buscar_manual_stream.py
import os
import sys
import time
import json
import requests
import torch
import concurrent.futures
import chromadb
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# Configurar encoding UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "db")

# Cliente de ChromaDB
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="manuales",
    metadata={"hnsw:space": "cosine", "hnsw:search_ef": 30}
)

# Variable global para el modelo de embeddings
embedder = None

def initialize_components():
    """Inicializa los componentes pesados al iniciar."""
    global embedder
    print("\n🔧 Inicializando modelo de embeddings...")
    start = time.time()
    # Cargar modelo en GPU con half-precision
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda").half()
    # Warm-up para forzar la carga en memoria
    with torch.no_grad():
        embedder.encode("warm-up", convert_to_tensor=True, device="cuda")
    print(f"✅ Modelo listo en {time.time() - start:.2f}s\n")

@lru_cache(maxsize=1000)
def generate_embedding(query: str):
    """Genera y cachea la representación (embedding) de la consulta."""
    return embedder.encode(query, convert_to_tensor=True).half().cpu().numpy()

def ollama_generate_response(query, context):
    """
    Generador de respuestas que utiliza streaming desde el modelo externo.
    Cada fragmento se yield y se procesa en formato texto.
    """
    MAX_CONTEXT_LENGTH = 2000
    truncated_context = context[:MAX_CONTEXT_LENGTH]
    
    payload = {
        "model": "mistral",
        "prompt": f"""[INST] 
Eres un asistente técnico experto y solo puedes responder con información que esté en el contexto.
Si la respuesta no está explícitamente en el contexto, responde exactamente: "No se encontró información en los manuales." No intentes adivinar ni generar información adicional.

### Contexto:
{context}

### Pregunta:
{query}

### Respuesta:
[/INST]""",
        "stream": True,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
            "repeat_penalty": 1.2,
            "num_thread": 8
        }
    }
    
    try:
        with requests.post("http://127.0.0.1:11434/api/generate", json=payload, stream=True) as response:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        yield chunk
                    except json.JSONDecodeError:
                        yield line
    except Exception as e:
        yield f"Error: {str(e)}"

def search_manual(query: str, top_k: int = 3):
    """
    Realiza la búsqueda en la base y retorna un generador que transmite la respuesta
    en streaming. Cada fragmento se formatea en JSON para facilitar su interpretación
    en el cliente.
    """
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embed_future = executor.submit(generate_embedding, query)
        chroma_future = executor.submit(
            lambda: collection.query(
                query_embeddings=[embed_future.result().tolist()],
                n_results=top_k
            )
        )
        results = chroma_future.result()
    
    if not results or not results.get("documents"):
        yield json.dumps({"respuesta": "No se encontró información relevante"}) + "\n"
        return

    # Concatenar los documentos obtenidos para construir el contexto
    context = "\n".join([doc for sublist in results["documents"] for doc in sublist])
    
    # Transmitir la respuesta en streaming, fragmento a fragmento, formateada en JSON
    for chunk in ollama_generate_response(query, context):
        yield json.dumps({"respuesta": chunk}) + "\n"
    
    print(f"Tiempos [Búsqueda: {time.time() - start_time:.2f}s]")

if __name__ == "__main__":
    initialize_components()
    print("Sistema de búsqueda de manuales. Escribe 'salir' para terminar.")
    while True:
        query = input("Ingresa tu consulta: ")
        if query.lower() == "salir":
            break
        for respuesta in search_manual(query):
            print(respuesta, end="")
