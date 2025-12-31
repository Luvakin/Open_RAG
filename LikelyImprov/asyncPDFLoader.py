import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from loader import PDFLoader
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import pymupdf
import pdf2image
import pytesseract


@dataclass
class PageResult:
    page_number: int
    text: str
    page_type:str
    source: str
    error: Optional[str] =None


@dataclass
class FileResult:
    file_path: str
    pages: List[PageResult]
    total_text: str
    pdf_type: str
    error: Optional[str] = None

class AsyncMultiPDFLoader:

    def __init__(self, max_workers: int = 4, max_page_workers: int = 8):
        self.fileExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self.pageExecutor = ThreadPoolExecutor(max_workers=max_page_workers)

    
    async def process_files_async(self, filePaths: List[str]) -> Dict[str, FileResult]:
        'To process multiple PDF file asynchronously'
        print(f'Processing {len(filePaths)} PDF files asynchronously...')

        # Create a task for each file
        tasks = [
            self.process_single_file_async(filePath)
            for filePath in filePaths
        ]

        # Wait for all tasks to Complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        fileResults = {}
        for filePath, result in zip(filePaths, results):
            if isinstance(result, Exception):
                fileResults[filePath] = FileResult(
                    file_path=filePath,
                    pages=[],
                    total_text='',
                    pdf_type='',
                    error = str(result))
                
            else:
                fileResults[filePath] = result
        
        return fileResults
    
    async def process_single_file_async(self, filePath: str) -> FileResult:
        'To Process a Single PDF file asynchronously'
        print('Processing File:', Path(filePath).name)

        if not os.path.exists(filePath):
            return FileResult(
                file_path=filePath,
                pages=[],
                total_text='',
                pdf_type='Error',
                error=f'File Not Found: {filePath}'
            )
        
        try:
            # Detect PDF type asynchronously
            pdf_type= await self._detect_pdf_type_async(filePath)
            print(f'Detected PDF type for {Path(filePath).name} as {pdf_type}')

            # Process of the of Pages based on PDF type
            if pdf_type == 'Text-based PDF':
                pages_results = await self._process_text_pages_async(filePath)
            elif pdf_type == 'Image-based (Scanned) PDF':
                pages_results = await self._process_image_pages_async(filePath)
            else:
                pages_results = []

            total_text = ''.join([page.text for page in pages_results if page.text])
            return FileResult(
                file_path=filePath,
                pages=pages_results,
                total_text=total_text,
                pdf_type=pdf_type,
                error=None
            )
        except Exception as e:
            print(f'Error processing file:, {filePath}: {e}')
            return FileResult(
                file_path=filePath,
                pages=[],
                total_text='',
                pdf_type='Error',
                error=str(e)
            )
        
    async def _detect_pdf_type_async(self, filePath: str) -> str:
        'To detect the type of PDF file async'
        loop = asyncio.get_event_loop()

        def detect_pdf_type():
            try:
                with pymupdf.open(filePath) as docs:
                    total_text = 0
                    for page in docs[:3]:
                        text = page.get_text().strip()
                        total_text += len(text)
                        if total_text > 0:
                            return 'Text-based PDF'
                    
                    return 'Text-based PDF' if total_text != 0 else 'Image-based (Scanned) PDF'
                
            except Exception:
                return 'Unknown PDF Type'
            
        return await loop.run_in_executor(self.fileExecutor, detect_pdf_type)
            
    async def _process_text_pages_async(self, filePath:str) -> List[PageResult]:
        """Processing of Text-based PDF files Asynchronously"""

        loop = asyncio.get_event_loop()

        def get_page_count():
            try: 
                with pymupdf.open(filePath) as docs:
                    return len(docs)
                
            except Exception as e:
                print(f'Error getting page count for {filePath}: {e}')
                return 0
        
        page_count = await loop.run_in_executor(self.fileExecutor, get_page_count)
        print(f'Processing {page_count} pages in {Path(filePath).name}...')

        # Create tasks for each page
        tasks = [
            self._extract_text_page_async(filePath, page_num)
            for page_num in range(page_count)
        ]

        page_results = await asyncio.gather(*tasks, return_exceptions=True)


        # Handle Exceptions
        final_results = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                final_results.append(PageResult(
                    page_number = i + 1,
                    text='',
                    page_type='text',
                    source=filePath,
                    error = str(result)
                ))
            else:
                final_results.append(result)

        return final_results
    
    async def _extract_text_page_async(self, filePath:str, page_num:int) -> PageResult:
        """Extract text from a single page asynchronously"""
        loop = asyncio.get_event_loop()

        def extract_text():
            try:
                with pymupdf.open(filePath) as docs:
                    page = docs[page_num]
                    text = page.get_text().strip()

                    return PageResult(
                        page_number = page_num + 1,
                        text = text,
                        page_type = 'text',
                        source =filePath,
                    )
                
            except Exception as e:
                print(f'Error extracting text from page {page_num + 1} of {filePath}: {e}')
                return PageResult(
                        page_number = page_num + 1,
                        text = text,
                        page_type = 'text',
                        source =filePath,
                        error = str(e)
                    )
            
        return await loop.run_in_executor(self.pageExecutor, extract_text)
    
    async def _process_image_pages_async(self, filePath: str) -> List[PageResult]:
        """Processing Imagebased PDF files async"""
        loop = asyncio.get_event_loop()

        def get_image_of_pages():
            from PyPDF2 import PdfReader
            try:
                return len(PdfReader(filePath).pages)

                #return  pdf2image.convert_from_path(filePath) 
            except Exception as e:
                print('Error Converting PDF to image:', e)
                return 0
            
        images = await loop.run_in_executor(self.fileExecutor, get_image_of_pages)
        print(f'Processing {len(images)} pages in {Path(filePath).name}...')

        # Create tasks for each image page
        tasks = [
            self._extract_ocr_page_async(filePath, i, image)
            for i, image in enumerate(images)
        ]

        page_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle Exceptions
        final_results = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                final_results.append(PageResult(
                    page_number= i + 1,
                    text = '',
                    page_type='ocr',
                    source=filePath,
                    error= str(result)
                ))
            else:
                final_results.append(result)

        return final_results
    
    async def _extract_ocr_page_async(self, filePath: str, page_num:int, image) -> PageResult:
        """Extract text from image pages using OCR asynchronously"""

        loop = asyncio.get_event_loop()

        def extract_ocr_text():
            try:

                text = pytesseract.image_to_string(image)
                return PageResult(
                    page_number=page_num +1,
                    text=text,
                    page_type='ocr',
                    source=filePath
                )
            
            except Exception as e:
                print(f'Error extracting OCR text from page {page_num + 1} of {filePath}: {e}')
                return PageResult(
                    page_number=page_num + 1,
                    text='',
                    page_type='ocr',
                    source=filePath,
                    error=str(e)
                )
            
        return await loop.run_in_executor(self.pageExecutor, extract_ocr_text)

    def close(self):
        """Close the executore"""
        self.fileExecutor.shutdown(wait=True)
        self.pageExecutor.shutdown(wait=True)

filePath = "C:/Users/Dell/Documents/RUN/400L/Hadi Saadat.pdf"
filePath1 = "C:/Users/Dell/Documents/RUN/400L/cpe_403_.pdf"
filePath2 = "C:/Users/Dell/Documents/Arinde David Shina-Ayomi Acceptance Letter_111924.pdf"
filepath3 = "C:/Users/Dell/Documents/COR.pdf"
filepath4 = "C:/Users/Dell/Documents/CO.pdf"
filePath5 = "C:/Users/Dell/Downloads/RLT.pdf"
filePath6 = "C:/Users/Dell/Downloads/Machine Learning Lecture Note.pdf"


filePaths = [filePath1, filePath2, filepath3, filepath4, filePath5, filePath6]


async def main():
    processor = AsyncMultiPDFLoader(max_workers=4, max_page_workers=8)

    existingFilePath = [filePath for filePath  in filePaths if os.path.exists(filePath)]

    if not existingFilePath:
        print("No valid PDF files found!")
        return
    
    print(f'{len(existingFilePath)} file found for processing...')

    # Process all files 
    startTime = asyncio.get_event_loop().time()
    results = await processor.process_files_async(existingFilePath)
    endTime = asyncio.get_event_loop().time()

    print(f'Processing Time: {endTime - startTime:.2f} secs')

    for file_path, result in results.items():
        print(f"\nFile: {Path(file_path).name}")
        print(f"  Type: {result.pdf_type}")
        print(f"  Pages: {len(result.pages)}")
        print(f"  Total text: {len(result.total_text)} characters")
        
        if result.error:
            print(f"  Error: {result.error}")
        else:
            # Show first few pages info
            for page in result.pages[:3]:
                status = "✓" if not page.error else "✗"
                print(f"  Page {page.page_number} ({page.page_type}): {status} {len(page.text)} chars")
            
            if len(result.pages) > 3:
                print(f"  ... and {len(result.pages) - 3} more pages")
            
            # Show text preview
            if result.total_text:
                print(f"  Preview: {result.total_text[:100]}...")
    
    # Clean up
    processor.close()
   



if __name__ == '__main__':
    asyncio.run(main())