import chromadb

def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="manuales")
    return collection
