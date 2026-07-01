import chromadb

class VectorDB:

    def __init__(self, del_coll : bool = False) -> None:
        #self.client = chromadb.Client()
        self.client = chromadb.PersistentClient(path = './DB')
        self.client.delete_collection(name = 'IMG_Dataset') if del_coll else None
        self.collection = self.client.get_or_create_collection(name = 'IMG_Dataset')
    

    def insert_embedding(self, embeddings, documents, metadatas, ids):
        self.collection.add(
            embeddings = embeddings,
            documents = documents,
            metadatas = metadatas,
            ids = ids
        )

    def query_image(self, vector_embeddings, n_results):
        response = self.collection.query(
            query_embeddings = vector_embeddings,
            n_results = n_results
        )
        return response