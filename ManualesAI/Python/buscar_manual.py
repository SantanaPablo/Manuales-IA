# buscar_maual.py
import os
import sys
import requests
import time
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import concurrent.futures

# Configurar encoding UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "db")

# Client ChromaDB
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(
    name="manuales",
    metadata={"hnsw:space": "cosine", "hnsw:search_ef": 30}
)

# Variables globales
embedder = None

def initialize_components():
    """Inicializa los componentes pesados al iniciar"""
    global embedder
    
    print("\n🔧 Inicializando modelo de embeddings...")
    start = time.time()
    
    # Cargar modelo en GPU con half-precision
    embedder = SentenceTransformer("BAAI/bge-large-en", device="cuda").half()
    
    # Warm-up inicial para forzar carga en memoria
    with torch.no_grad():
        embedder.encode("warm-up", convert_to_tensor=True, device="cuda")
    
    print(f"✅ Modelo listo en {time.time() - start:.2f}s\n")

@lru_cache(maxsize=1000)
def generate_embedding(query: str):
    return embedder.encode(query, convert_to_tensor=True).half().cpu().numpy()

def ollama_generate_response(query, context):
    """Generador de respuestas optimizado"""
    MAX_CONTEXT_LENGTH = 2000
    truncated_context = context[:MAX_CONTEXT_LENGTH]
    
    payload = {
        "model": "mistral",
        "prompt": f"[INST]{truncated_context}\n\nPregunta: {query} [/INST]",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
            "repeat_penalty": 1.2,
            "num_thread": 8
        }
    }
    
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {str(e)}"

def search_manual(query: str, top_k: int = 3):
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
        return "No se encontró información relevante"

    context = "\n".join([doc for sublist in results["documents"] for doc in sublist])
    respuesta = ollama_generate_response(query, context)
    
    print(f"Tiempos [Búsqueda: {time.time() - start_time:.2f}s]")
    return respuesta