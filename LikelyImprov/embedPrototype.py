import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
langchainApiKey = os.getenv('LANGCHAIN_API_KEY')

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'default'
os.environ['LANGCHAIN_API_KEY'] = langchainApiKey
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'



class Embedding:
    def __init__(self, embeddingModel=OllamaEmbeddings(model='nomic-embed-text'), chunk_size = 300, chunk_overlap = 50, path=os.getcwd()):
        self.embeddingModel = embeddingModel
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.path = path

    
    def textSplit(self, pages):
    
        try:
            spliter = RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
            )

            splits = spliter.split_documents(pages)


        except Exception as e:
            print(f'Error occured during spilting: {e}')

        
        valid_spilts = [doc for doc in splits if doc.page_content.strip()]

        if len(valid_spilts) < len(splits):
            print(f'Warning: {len(splits) - len(valid_spilts)} empty splits')

        return valid_spilts
    


    def embedding_and_vectorstore(self, valid_splits, saveVectorStore = True, knn = 1):

        embeddingModel = self.embeddingModel
        try:
            test_embed = embeddingModel.embed_query("Test document")
            print(f"Embedding test successful, dimension: {len(test_embed)}")
        except Exception as e:
            print(f"Embedding test failed: {str(e)}")
            raise

        try:
            vectorStore = Chroma.from_documents(
                documents = valid_splits,
                embedding=embeddingModel,
                persist_directory=f'./chroma_{Path(self.path).stem}.db'
            )

            if saveVectorStore:
                vectorStore.persist()

        except Exception as e:
            print(f'Error while storing docs: {e}')

        
        retriever = vectorStore.as_retriever(search_kwargs = {'k':knn})

        return retriever