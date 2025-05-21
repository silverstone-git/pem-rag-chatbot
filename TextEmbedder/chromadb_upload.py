import chromadb

def add(collection_name: str, docs: list[str], ids: list[str]):
    chroma_client = chromadb.PersistentClient()
    coll= chroma_client.get_or_create_collection(name= collection_name)
    coll.add(documents= docs, ids= ids)

if __name__ == "__main__":
    print("for adding docs to a collection")
