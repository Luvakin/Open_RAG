import asyncio
from langchain_community.document_loaders import UnstructuredFileLoader
from pathlib import Path

async def load_document_async(file_path: str):
    """Load a single document asynchronously"""
    def _load_sync():
        loader = UnstructuredFileLoader(file_path=file_path, strategy = 'ocr_only')
        return loader.load()
    
    return await asyncio.to_thread(_load_sync)

async def load_documents_batch(file_paths: list[str], max_concurrent: int = 5):
    """Load multiple documents with concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def load_with_semaphore(file_path):
        async with semaphore:
            return await load_document_async(file_path)
    
    tasks = [load_with_semaphore(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and flatten results
    documents = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Error loading document: {result}")
        else:
            documents.extend(result)
    
    return documents

# Usage
async def main():
    filePath = "C:/Users/Dell/Documents/RUN/400L/Hadi Saadat.pdf"
    filepath3 = "C:/Users/Dell/Documents/COR.pdf"
    file_paths = [filePath, filepath3]
    documents = await load_documents_batch(file_paths)
    print(f"Loaded {len(documents)} documents")

# Run it
asyncio.run(main())