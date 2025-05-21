import chromadb

chroma_client = chromadb.PersistentClient('/home/cyto/dev/pem-rag-chatbot/chroma')

def add(collection_name: str, docs: list[str], ids: list[str]):
    coll= chroma_client.get_or_create_collection(name= collection_name)
    coll.add(documents= docs, ids= ids)

# Assume this helper function exists and returns a list of dictionaries
# like [{ 'id': 'chunk_id', 'document': 'chunk string' }]
def querydb(collection_name, query, n_results=5):
    """
    Helper function to query ChromaDB and return relevant chunks.
    """
    # print("query for db: ")
    # print(query)
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        results = collection.query(
            query_texts= [query],
            n_results=n_results,
        )
        
        # print("results: ", results)

        relevant_chunks = []

        if results and results.get('documents'):
            for i in range(len(results['documents'][0])):
                # Assuming 'documents' is a list of lists where each inner list contains chunk strings
                relevant_chunks.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i]
                })
        return relevant_chunks

    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []


if __name__ == "__main__":
    print("for adding docs to a collection")
