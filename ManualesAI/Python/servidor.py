# servidor.py
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Importa el middleware de CORS
from buscar_manual_stream import search_manual, initialize_components

app = FastAPI()

# Agregar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes desde cualquier origen; para producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def initialize_services():
    """Inicializa los componentes en un hilo separado al arrancar."""
    print("🚀 Iniciando servicios backend...")
    init_thread = threading.Thread(target=initialize_components)
    init_thread.start()
    init_thread.join()
    print("🏁 Servicios listos para recibir solicitudes")

@app.get("/")
def home():
    return {"status": "active", "service": "buscar_manual"}

@app.get("/buscar/")
def buscar(pregunta: str):
    """
    Endpoint que recibe una pregunta y retorna la respuesta en streaming.
    La respuesta se envía como líneas de JSON.
    """
    generator = search_manual(pregunta)
    return StreamingResponse(generator, media_type="application/json")
