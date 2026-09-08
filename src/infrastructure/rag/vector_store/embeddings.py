import logging
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.core.config import settings
import os

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.encoder = None
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded")
            
            db_path = os.path.join(settings.DATA_DIR, "vector_db")
            os.makedirs(db_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=db_path)
            
            self.collection = self.client.get_or_create_collection(
                name="currency_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Vector store initialization error: {e}")
            self.encoder = None
            self.client = None
            self.collection = None
    
    def index_documents(self, documents: List[Dict]) -> bool:
        if not self.collection or not self.encoder:
            logger.error("Vector store not initialized")
            return False
        
        try:
            if self.collection.count() > 0:
                self.collection.delete(ids=[str(i) for i in range(self.collection.count())])
            
            texts = [doc['text'] for doc in documents]
            metadatas = [{'source': doc.get('source', 'unknown')} for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            embeddings = self.encoder.encode(texts).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return False
    
    def search(self, query: str, n_results: int = 3) -> List[str]:
        if not self.collection or not self.encoder:
            logger.error("Vector store not initialized")
            return []
        
        try:
            query_embedding = self.encoder.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def count(self) -> int:
        if not self.collection:
            return 0
        return self.collection.count()
    
    def clear(self) -> bool:
        if not self.collection:
            return False
        try:
            if self.collection.count() > 0:
                self.collection.delete(ids=[str(i) for i in range(self.collection.count())])
            logger.info("Vector store cleared")
            return True
        except Exception as e:
            logger.error(f"Clear error: {e}")
            return False
