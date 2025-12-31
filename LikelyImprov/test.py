import asyncio
import pymupdf
import pdf2image
import pytesseract
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Optional
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PageResult:
    page_number: int
    text: str
    page_type: str
    source: str
    error: Optional[str] = None

@dataclass
class FileResult:
    file_path: str
    pages: List[PageResult]
    total_text: str
    pdf_type: str
    error: Optional[str] = None

class AsyncMultiPdfProcessor:
    def __init__(self, max_workers: int = 4, max_page_workers: int = 8):
        self.file_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.page_executor = ThreadPoolExecutor(max_workers=max_page_workers)
    
    async def process_files_async(self, file_paths: List[str]) -> Dict[str, FileResult]:
        """Process multiple PDF files asynchronously"""
        print(f"Processing {len(file_paths)} files concurrently...")
        
        # Create tasks for each file
        tasks = [
            self.process_single_file_async(file_path) 
            for file_path in file_paths
        ]
        
        # Wait for all files to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return results mapped to file paths
        file_results = {}
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                file_results[file_path] = FileResult(
                    file_path=file_path,
                    pages=[],
                    total_text='',
                    pdf_type='Error',
                    error=str(result)
                )
            else:
                file_results[file_path] = result
        
        return file_results
    
    async def process_single_file_async(self, file_path: str) -> FileResult:
        """Process a single PDF file with async page processing"""
        print(f"Starting file: {Path(file_path).name}")
        
        if not os.path.exists(file_path):
            return FileResult(
                file_path=file_path,
                pages=[],
                total_text='',
                pdf_type='Error',
                error='File not found'
            )
        
        try:
            # Detect PDF type
            pdf_type = await self._detect_pdf_type_async(file_path)
            print(f"File {Path(file_path).name}: Detected as {pdf_type}")
            
            # Process pages based on type
            if pdf_type == 'Text-based PDF':
                page_results = await self._process_text_pages_async(file_path)
            elif pdf_type == 'Image-Based (Scanned) PDF':
                page_results = await self._process_image_pages_async(file_path)
            else:
                page_results = []
            
            # Combine all text
            total_text = '\n'.join([page.text for page in page_results if page.text])
            
            print(f"File {Path(file_path).name}: Completed - {len(page_results)} pages, {len(total_text)} characters")
            
            return FileResult(
                file_path=file_path,
                pages=page_results,
                total_text=total_text,
                pdf_type=pdf_type
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return FileResult(
                file_path=file_path,
                pages=[],
                total_text='',
                pdf_type='Error',
                error=str(e)
            )
    
    async def _detect_pdf_type_async(self, file_path: str) -> str:
        """Detect PDF type asynchronously"""
        loop = asyncio.get_event_loop()
        
        def detect_sync():
            try:
                with pymupdf.open(file_path) as doc:
                    total_text_length = 0
                    for page in doc[:3]:  # Check first 3 pages
                        text = page.get_text().strip()
                        total_text_length += len(text)
                        if total_text_length > 50:
                            return 'Text-based PDF'
                    
                    return 'Text-based PDF' if total_text_length > 10 else 'Image-Based (Scanned) PDF'
            except Exception:
                return 'Unknown PDF Type'
        
        return await loop.run_in_executor(self.file_executor, detect_sync)
    
    async def _process_text_pages_async(self, file_path: str) -> List[PageResult]:
        """Process text-based PDF pages asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Get page count first
        def get_page_count():
            with pymupdf.open(file_path) as doc:
                return len(doc)
        
        page_count = await loop.run_in_executor(self.file_executor, get_page_count)
        print(f"Processing {page_count} text pages for {Path(file_path).name}")
        
        # Create tasks for each page
        tasks = [
            self._extract_text_page_async(file_path, page_num)
            for page_num in range(page_count)
        ]
        
        # Process pages concurrently
        page_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                final_results.append(PageResult(
                    page_number=i + 1,
                    text='',
                    page_type='text',
                    source=file_path,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _extract_text_page_async(self, file_path: str, page_num: int) -> PageResult:
        """Extract text from a single page asynchronously"""
        loop = asyncio.get_event_loop()
        
        def extract_page_text():
            try:
                with pymupdf.open(file_path) as doc:
                    page = doc[page_num]
                    text = page.get_text().strip()
                    
                    # Try alternative extraction methods if no text found
                    if not text:
                        text = page.get_text("text").strip()
                    
                    if not text:
                        text_blocks = page.get_text("blocks")
                        text = " ".join([block[4] for block in text_blocks if len(block) > 4]).strip()
                    
                    return PageResult(
                        page_number=page_num + 1,
                        text=text,
                        page_type='text',
                        source=file_path
                    )
            except Exception as e:
                return PageResult(
                    page_number=page_num + 1,
                    text='',
                    page_type='text',
                    source=file_path,
                    error=str(e)
                )
        
        return await loop.run_in_executor(self.page_executor, extract_page_text)
    
    async def _process_image_pages_async(self, file_path: str) -> List[PageResult]:
        """Process image-based PDF pages asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Convert PDF to images first
        def convert_to_images():
            return pdf2image.convert_from_path(file_path)
        
        images = await loop.run_in_executor(self.file_executor, convert_to_images)
        print(f"Processing {len(images)} image pages for {Path(file_path).name}")
        
        # Create tasks for OCR processing
        tasks = [
            self._extract_ocr_page_async(file_path, i, image)
            for i, image in enumerate(images)
        ]
        
        # Process OCR concurrently
        page_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(page_results):
            if isinstance(result, Exception):
                final_results.append(PageResult(
                    page_number=i + 1,
                    text='',
                    page_type='ocr',
                    source=file_path,
                    error=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _extract_ocr_page_async(self, file_path: str, page_num: int, image) -> PageResult:
        """Extract text from a single image page using OCR asynchronously"""
        loop = asyncio.get_event_loop()
        
        def extract_ocr_text():
            try:
                ocr_text = pytesseract.image_to_string(image)
                return PageResult(
                    page_number=page_num + 1,
                    text=ocr_text,
                    page_type='ocr',
                    source=file_path
                )
            except Exception as e:
                return PageResult(
                    page_number=page_num + 1,
                    text='',
                    page_type='ocr',
                    source=file_path,
                    error=str(e)
                )
        
        return await loop.run_in_executor(self.page_executor, extract_ocr_text)
    
    def close(self):
        """Clean up thread pools"""
        self.file_executor.shutdown(wait=True)
        self.page_executor.shutdown(wait=True)

# Usage example
async def main():
    # Initialize processor
    processor = AsyncMultiPdfProcessor(
        max_workers=4,      # Files processed concurrently
        max_page_workers=8  # Pages processed concurrently per file
    )
    
    # List of PDF files to process
    file_paths = [
        "C:/Users/Dell/Documents/RUN/400L/Hadi Saadat.pdf",
        "C:/Users/Dell/Documents/RUN/400L/cpe_403_.pdf",
        # Add more files as needed
    ]
    
    # Filter existing files
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        print("No valid PDF files found!")
        return
    
    print(f"Found {len(existing_files)} PDF files to process")
    
    # Process all files
    start_time = asyncio.get_event_loop().time()
    results = await processor.process_files_async(existing_files)
    end_time = asyncio.get_event_loop().time()
    
    # Display results
    print(f"\n=== Processing completed in {end_time - start_time:.2f} seconds ===")
    
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

# Advanced usage: Process with progress tracking
async def process_with_progress():
    processor = AsyncMultiPdfProcessor(max_workers=2, max_page_workers=4)
    
    file_paths = [
        "C:/Users/Dell/Documents/RUN/400L/Hadi Saadat.pdf",
        "C:/Users/Dell/Documents/RUN/400L/cpe_403_.pdf",
    ]
    
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if existing_files:
        print("Processing with progress tracking...")
        
        # Create progress tracking
        tasks = [
            processor.process_single_file_async(file_path)
            for file_path in existing_files
        ]
        
        # Process with progress updates
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            print(f"Progress: {completed}/{len(existing_files)} files completed")
            print(f"  Latest: {Path(result.file_path).name} - {len(result.pages)} pages")
    
    processor.close()

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Or run with progress tracking
    # asyncio.run(process_with_progress())