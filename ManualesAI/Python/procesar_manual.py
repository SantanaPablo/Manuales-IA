# procesar_manual.py
import os
import time
import chromadb
from sentence_transformers import SentenceTransformer
import concurrent.futures
import re
import fitz  # PyMuPDF para PDF
import docx  # python-docx para Word
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer
import nltk

# Descargar el modelo de tokenización de NLTK (punkt) si aún no está disponible
nltk.download('punkt')

MANUALS_FOLDER = "manuales/"

# Configuración optimizada de ChromaDB
chroma_client = chromadb.PersistentClient(path="db")
collection = chroma_client.get_or_create_collection(
    name="manuales",
    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 50}
)

# Cargar modelo de embeddings en half-precision para mayor velocidad
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda").half()
# Inicializar AutoTokenizer para segmentación, usando el mismo modelo
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def file_already_indexed(filename):
    """Verifica si el archivo ya está indexado en ChromaDB."""
    existing_docs = collection.get(ids=[f"{filename}_0"])
    return len(existing_docs["ids"]) > 0

def read_text_file(file_path):
    """Lee archivos de texto, Word (.docx), PDF (.pdf) y Excel (.xls/.xlsx)."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        
        elif file_path.endswith(".docx"):
            return read_docx(file_path)
        
        elif file_path.endswith(".pdf"):
            return read_pdf(file_path)

        elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
            return read_xls(file_path)

    except Exception as e:
        print(f"❌ Error al leer {file_path}: {e}")
        return None

def read_docx(file_path):
    """Extrae texto de un archivo Word (.docx)."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        print(f"❌ Error al leer .docx {file_path}: {e}")
        return None

def read_xls(file_path):
    """Extrae texto de un archivo Excel (.xls o .xlsx)."""
    try:
        df = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")  # Leer todas las hojas
        text = []
        for sheet_name, sheet_data in df.items():
            text.append(f"--- {sheet_name} ---")  # Nombre de la hoja
            text.append(sheet_data.to_string(index=False, header=True))  # Convertir en texto
        return "\n".join(text).strip()
    except Exception as e:
        print(f"❌ Error al leer .xls/.xlsx {file_path}: {e}")
        return None

def read_pdf(file_path):
    """Extrae texto de un archivo PDF (.pdf)."""
    try:
        text = []
        with fitz.open(file_path) as doc:
            for page in doc:
                text.append(page.get_text("text"))
                print(f"xls indexado")
        return "\n".join(text).strip()
    except Exception as e:
        print(f"❌ Error al leer .pdf {file_path}: {e}")
        return None

def extract_metadata(segment):
    """
    Extrae metadata de un bloque de texto en formato estándar o intenta deducirla.
    Se elimina la conversión a minúsculas para preservar términos técnicos.
    """
    title_match = re.search(r"\[titulo=(.*?)\]", segment, re.IGNORECASE)
    info_match = re.search(r"\[información=(.*?)\]", segment, re.IGNORECASE | re.DOTALL)
    tags_match = re.search(r"\[etiquetas=(.*?)\]", segment, re.IGNORECASE)

    titulo = title_match.group(1).strip() if title_match else None
    informacion = info_match.group(1).strip() if info_match else None
    etiquetas = tags_match.group(1).strip().split(", ") if tags_match else []

    if not titulo or not informacion:  # Si falta metadata, intentar deducirla
        lines = segment.split("\n")
        # Tomar la primera línea en mayúsculas como título
        for line in lines:
            if len(line) < 50 and line.isupper():
                titulo = titulo or line.strip()
                break
        informacion = informacion or "\n".join(lines).strip()
        # Generar etiquetas a partir de palabras clave más frecuentes
        palabras = re.findall(r'\b[a-zA-Záéíóúüñ]{4,}\b', informacion)
        etiquetas = etiquetas or [w[0] for w in Counter(palabras).most_common(5)]
    
    return titulo or "Sin título", informacion or "", etiquetas

def segment_text(text, max_tokens=256, stride=128):
    """
    Segmenta el texto en fragmentos utilizando tokenización y ventanas deslizantes
    para evitar cortar el contexto. Se utiliza el AutoTokenizer para garantizar
    que los cortes se realicen en límites de tokens.
    """
    # Dividir el texto en oraciones usando NLTK
    sentences = nltk.sent_tokenize(text)
    segments = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        # Contar tokens en la oración actual
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        token_count = len(sentence_tokens)

        if token_count > max_tokens:
            # Si la oración es muy larga, segmentarla usando ventana deslizante en tokens
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            for i in range(0, len(token_ids), stride):
                window_token_ids = token_ids[i: i + max_tokens]
                segment_text_window = tokenizer.decode(window_token_ids, skip_special_tokens=True)
                segments.append(segment_text_window)
            continue

        if current_token_count + token_count <= max_tokens:
            current_sentences.append(sentence)
            current_token_count += token_count
        else:
            # Se crea un segmento a partir de las oraciones acumuladas
            segment = " ".join(current_sentences)
            segments.append(segment)
            # Ventana deslizante: conservar parte del contexto previo (solapamiento)
            overlap_sentences = []
            overlap_token_count = 0
            for s in reversed(current_sentences):
                s_tokens = tokenizer.encode(s, add_special_tokens=False)
                if overlap_token_count + len(s_tokens) <= stride:
                    overlap_sentences.insert(0, s)
                    overlap_token_count += len(s_tokens)
                else:
                    break
            current_sentences = overlap_sentences + [sentence]
            current_token_count = overlap_token_count + token_count

    if current_sentences:
        segments.append(" ".join(current_sentences))

    return segments

def process_file(file_path):
    """Procesa un archivo y lo indexa en ChromaDB."""
    filename = os.path.basename(file_path)
    
    if not file_path.endswith((".txt", ".docx", ".pdf", ".xlsx")):
        return
    
    if file_already_indexed(filename):
        print(f"⏩ {filename} ya está indexado, se omite.")
        return

    text = read_text_file(file_path)
    if not text:
        return
    
    # Separar en bloques (por ejemplo, párrafos separados por doble salto de línea)
    segments = text.split("\n\n")
    data = [extract_metadata(seg) for seg in segments]

    processed_data = []
    for titulo, informacion, etiquetas in data:
        # Aplicar la segmentación mejorada con ventanas deslizantes
        subsegments = segment_text(informacion)
        for subseg in subsegments:
            processed_data.append((titulo, subseg, etiquetas))

    # Generar embeddings en batch (paralelizando la codificación de cada segmento)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(
            lambda x: embedder.encode(x[1], convert_to_tensor=True).half().cpu().tolist(),
            processed_data
        ))

    collection.add(
        documents=[d[1] for d in processed_data],  # Documentos segmentados
        metadatas=[{"filename": filename, "titulo": d[0]} for d in processed_data],
        ids=[f"{filename}_{i}" for i in range(len(processed_data))],
        embeddings=embeddings
    )
    print(f"✅ {filename} indexado ({len(processed_data)} segmentos)")

def process_manuals():
    """Procesa todos los manuales en la carpeta."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in os.listdir(MANUALS_FOLDER):
            file_path = os.path.join(MANUALS_FOLDER, file)
            if file.endswith((".txt", ".docx", ".pdf", ".xlsx")):
                futures.append(executor.submit(process_file, file_path))
        
        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    start = time.time()
    process_manuals()
    print(f"Indexación completa en {time.time() - start:.2f} segundos")
