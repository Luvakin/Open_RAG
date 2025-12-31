import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from joblib import Memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize cache
memory = Memory(location='./embedding_cache', verbose=0)

class Embedding:
    def __init__(self, embeddingModel=OllamaEmbeddings(model='nomic-embed-text'), chunk_size=300, chunk_overlap=50, path=os.getcwd()):
        self.embeddingModel = embeddingModel
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.path = path
        logger.info(f"Initialized Embedding class with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def set_embedding_model(self, model_type: str, model_name: str = None, **kwargs):
        try:
            if model_type.lower() == 'ollama':
                self.embeddingModel = OllamaEmbeddings(model=model_name or 'nomic-embed-text', **kwargs)
            elif model_type.lower() == 'huggingface':
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddingModel = HuggingFaceEmbeddings(model_name=model_name or 'sentence-transformers/all-MiniLM-L6-v2', **kwargs)
            elif model_type.lower() == 'openai':
                from langchain_openai import OpenAIEmbeddings
                self.embeddingModel = OpenAIEmbeddings(model=model_name or 'text-embedding-ada-002', **kwargs)
            else:
                raise ValueError(f"Unsupported embedding model type: {model_type}")
            test_embed = self.embeddingModel.embed_query("Test document")
            logger.info(f"Switched to {model_type} embedding model, dimension: {len(test_embed)}")
        except Exception as e:
            logger.error(f"Failed to set embedding model: {str(e)}")
            raise

    def configure_text_splitter(self, splitter_type: str = 'recursive', chunk_size: int = None, chunk_overlap: int = None, **kwargs):
        try:
            chunk_size = chunk_size or self.chunk_size
            chunk_overlap = chunk_overlap or self.chunk_overlap
            if splitter_type.lower() == 'recursive':
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **kwargs
                )
            elif splitter_type.lower() == 'token':
                self.text_splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported splitter type: {splitter_type}")
            logger.info(f"Text splitter configured: {splitter_type} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
        except Exception as e:
            logger.error(f"Failed to configure text splitter: {str(e)}")
            raise

    def validate_inputs(self, pages):
        if not pages:
            raise ValueError("No documents provided for splitting")
        if not all(hasattr(doc, 'page_content') for doc in pages):
            raise ValueError("Invalid document format: page_content missing")
        logger.info("Input validation passed")

    @memory.cache
    def compute_embedding(self, text: str):
        return self.embeddingModel.embed_query(text)

    def embed_documents_parallel(self, documents, max_workers: int = 4):
        def embed_single(doc):
            return self.compute_embedding(doc.page_content)
        
        try:
            embeddings = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {executor.submit(embed_single, doc): doc for doc in documents}
                for future in as_completed(future_to_doc):
                    embeddings.append(future.result())
            return embeddings
        except Exception as e:
            logger.error(f"Error during parallel embedding: {str(e)}")
            raise

    def textSplit(self, pages, enrich_metadata: dict = None):
        logger.info("Starting text splitting")
        try:
            self.validate_inputs(pages)
            if not hasattr(self, 'text_splitter'):
                self.configure_text_splitter()
            splits = self.text_splitter.split_documents(pages)
            
            if enrich_metadata:
                for split in splits:
                    split.metadata.update(enrich_metadata)
            
            valid_splits = [doc for doc in splits if doc.page_content.strip()]
            if len(valid_splits) < len(splits):
                logger.warning(f"{len(splits) - len(valid_splits)} empty splits detected")
            logger.info(f"Text splitting completed: {len(valid_splits)} valid splits")
            return valid_splits
        except Exception as e:
            logger.error(f"Error during splitting: {str(e)}")
            return []

    def configure_retriever(self, vectorStore, retriever_type: str = 'similarity', knn: int = 1, **kwargs):
        try:
            if retriever_type.lower() == 'similarity':
                return vectorStore.as_retriever(search_type='similarity', search_kwargs={'k': knn, **kwargs})
            elif retriever_type.lower() == 'mmr':
                return vectorStore.as_retriever(search_type='mmr', search_kwargs={'k': knn, **kwargs})
            else:
                raise ValueError(f"Unsupported retriever type: {retriever_type}")
        except Exception as e:
            logger.error(f"Error configuring retriever: {str(e)}")
            raise

    def load_or_create_vectorstore(self, valid_splits, saveVectorStore: bool = True, knn: int = 1, batch_size: int = 1000):
        persist_directory = f'./chroma_{Path(self.path).stem}.db'
        try:
            if os.path.exists(persist_directory):
                logger.info(f"Loading existing vector store from {persist_directory}")
                vectorStore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddingModel
                )
            else:
                logger.info(f"Creating new vector store at {persist_directory}")
                vectorStore = Chroma(embedding_function=self.embeddingModel, persist_directory=persist_directory if saveVectorStore else None)
                for i in range(0, len(valid_splits), batch_size):
                    batch = valid_splits[i:i + batch_size]
                    logger.info(f"Processing batch {i // batch_size + 1} of {len(valid_splits) // batch_size + 1}")
                    vectorStore.add_documents(documents=batch)
                if saveVectorStore:
                    vectorStore.persist()
            return self.configure_retriever(vectorStore, retriever_type='similarity', knn=knn)
        except Exception as e:
            logger.error(f"Error in vector store creation/loading: {str(e)}")
            raise

    def embedding_and_vectorstore(self, valid_splits, saveVectorStore=True, knn=1):
        try:
            test_embed = self.embeddingModel.embed_query("Test document")
            logger.info(f"Embedding test successful, dimension: {len(test_embed)}")
            return self.load_or_create_vectorstore(valid_splits, saveVectorStore, knn)
        except Exception as e:
            logger.error(f"Embedding test failed: {str(e)}")
            raise

    def cleanup_vectorstore(self):
        persist_directory = f'./chroma_{Path(self.path).stem}.db'
        try:
            if os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
                logger.info(f"Cleaned up vector store at {persist_directory}")
            else:
                logger.info(f"No vector store found at {persist_directory}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")