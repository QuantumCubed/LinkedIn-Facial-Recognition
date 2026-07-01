import chromadb
from chromadb.config import Settings

class VectorDB:

    def __init__(self) -> None:
        #self.client = chromadb.Client()
        self.client = chromadb.PersistentClient(path = './DB', settings = Settings(allow_reset = True))
        self.collection = self.client.get_or_create_collection(name = 'IMG_Dataset')
    

    def insert_embedding(self, embeddings, documents, metadatas, ids) -> None:
        self.collection.add(
            embeddings = embeddings,
            documents = documents,
            metadatas = metadatas,
            ids = ids
        )

    def query_image(self, vector_embeddings, n_results) -> chromadb.QueryResult:
        response = self.collection.query(
            query_embeddings = vector_embeddings,
            n_results = n_results
        )
        return response