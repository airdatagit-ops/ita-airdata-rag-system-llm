"""
OCR Processor for scanned PDFs using PaddleOCR.

Extracts text from PDF images when native text extraction fails.
Based on PDFOCRProcessor.py provided by the user.

Usage:
    from parsers.ocr_processor import OCRProcessor
    
    ocr = OCRProcessor()
    text = ocr.extract_text_from_pdf(pdf_bytes)
"""

import io
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
from PIL import Image
from loguru import logger

# OCR and PDF imports (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr paddlepaddle")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class OCRProcessor:
    """
    OCR Processor for extracting text from scanned PDFs.
    Uses PaddleOCR for Portuguese text recognition.
    """
    
    _instance = None
    _ocr = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid reinitializing OCR engine."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, use_gpu: bool = False, lang: str = 'pt'):
        """
        Initialize OCR processor.
        
        Args:
            use_gpu: Use GPU acceleration (requires CUDA)
            lang: Language for OCR ('pt' for Portuguese)
        """
        if OCRProcessor._ocr is not None:
            return  # Already initialized
            
        self.use_gpu = use_gpu
        self.lang = lang
        self._initialized = False
        
    def _init_ocr(self):
        """Lazy initialization of PaddleOCR engine."""
        if self._initialized or OCRProcessor._ocr is not None:
            return True
            
        if not PADDLE_AVAILABLE:
            logger.error("PaddleOCR not available")
            return False
            
        try:
            import logging
            logging.getLogger('ppocr').setLevel(logging.WARNING)

            logger.info("Initializing PaddleOCR engine (this may take a moment)...")
            OCRProcessor._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
            )
            self._initialized = True
            logger.success("PaddleOCR initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        if not self._init_ocr():
            return ""
            
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Run OCR using the ocr() method (PaddleOCR 2.x API)
            result = OCRProcessor._ocr.ocr(img_array, cls=True)
            
            # Extract text from result
            # PaddleOCR returns: [[[box, (text, confidence)], ...], ...]
            text_lines = []
            if result and len(result) > 0 and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        # line[1] is (text, confidence)
                        text_info = line[1]
                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                            text_lines.append(text_info[0])
                        elif isinstance(text_info, str):
                            text_lines.append(text_info)
            
            return "\n".join(text_lines)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, resolution: int = 300) -> Optional[str]:
        """
        Extract text from PDF bytes, using OCR for scanned pages.
        
        Args:
            pdf_bytes: PDF file content as bytes
            resolution: DPI for rendering pages (higher = better OCR, slower)
            
        Returns:
            Extracted text or None if failed
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.error("pdfplumber not available for PDF processing")
            return None
            
        try:
            all_text = []
            ocr_pages = 0
            native_pages = 0
            
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(f"Processing {total_pages} pages...")
                
                for i, page in enumerate(pdf.pages, start=1):
                    # Try native text extraction first
                    native_text = page.extract_text()
                    
                    if native_text and len(native_text.strip()) > 50:
                        # Native text is good enough
                        all_text.append(f"--- PÁGINA {i} ---\n{native_text}")
                        native_pages += 1
                    else:
                        # Need OCR
                        if not self._init_ocr():
                            # OCR not available, skip this page
                            logger.warning(f"Skipping page {i} - OCR not available")
                            continue
                            
                        logger.debug(f"Page {i}: using OCR...")
                        
                        # Convert page to high-resolution image
                        img = page.to_image(resolution=resolution).original
                        
                        # Extract text with OCR
                        ocr_text = self.extract_text_from_image(img)
                        
                        if ocr_text:
                            all_text.append(f"--- PÁGINA {i} ---\n{ocr_text}")
                            ocr_pages += 1
                        else:
                            logger.warning(f"No text extracted from page {i}")
            
            if all_text:
                logger.info(f"Extracted text: {native_pages} native pages, {ocr_pages} OCR pages")
                return "\n\n".join(all_text)
            
            return None
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return None
    
    def extract_text_from_pdf_file(self, pdf_path: Path, resolution: int = 300) -> Optional[str]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            resolution: DPI for rendering pages
            
        Returns:
            Extracted text or None if failed
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            return self.extract_text_from_pdf_bytes(pdf_bytes, resolution)
        except Exception as e:
            logger.error(f"Failed to read PDF file {pdf_path}: {e}")
            return None


class OCRBatchProcessor:
    """
    Batch processor for OCR extraction of multiple PDFs.
    Maintains a processing log to resume interrupted batches.
    """
    
    def __init__(self, input_dir: Path, output_dir: Path = None, use_gpu: bool = False):
        """
        Initialize batch processor.
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Directory for output text files (default: input_dir/ocr_output)
            use_gpu: Use GPU acceleration
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "ocr_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr = OCRProcessor(use_gpu=use_gpu)
        
        # Processing log
        self.log_file = self.output_dir / "processing_log.json"
        self.log = self._load_log()
    
    def _load_log(self) -> Dict:
        """Load processing log."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"processed": [], "failed": [], "last_run": None}
    
    def _save_log(self):
        """Save processing log."""
        self.log["last_run"] = datetime.now().isoformat()
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)
    
    def process_all(self, force: bool = False, limit: int = None) -> Dict:
        """
        Process all PDFs in the input directory.
        
        Args:
            force: Re-process already processed files
            limit: Maximum number of files to process
            
        Returns:
            Summary dict with results
        """
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if limit:
            pdf_files = pdf_files[:limit]
        
        logger.info(f"Processing {len(pdf_files)} PDFs from {self.input_dir}")
        
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for pdf_path in pdf_files:
            name = pdf_path.stem
            output_path = self.output_dir / f"{name}.txt"
            
            # Skip if already processed
            if not force and name in self.log["processed"] and output_path.exists():
                logger.debug(f"Skipping {name} (already processed)")
                results["skipped"] += 1
                continue
            
            logger.info(f"Processing: {name}")
            
            try:
                text = self.ocr.extract_text_from_pdf_file(pdf_path)
                
                if text:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    self.log["processed"].append(name)
                    results["success"] += 1
                    logger.success(f"Saved: {output_path.name} ({len(text)} chars)")
                else:
                    self.log["failed"].append({"file": name, "error": "No text extracted"})
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {name}: {e}")
                self.log["failed"].append({"file": name, "error": str(e)})
                results["failed"] += 1
            
            self._save_log()
        
        logger.info(f"Batch complete: {results['success']} success, {results['failed']} failed, {results['skipped']} skipped")
        return results


# Convenience function for quick OCR extraction
def extract_text_with_ocr(pdf_bytes_or_path, use_gpu: bool = False) -> Optional[str]:
    """
    Extract text from a PDF using OCR if needed.
    
    Args:
        pdf_bytes_or_path: PDF bytes or Path to PDF file
        use_gpu: Use GPU acceleration
        
    Returns:
        Extracted text or None
    """
    ocr = OCRProcessor(use_gpu=use_gpu)
    
    if isinstance(pdf_bytes_or_path, (str, Path)):
        return ocr.extract_text_from_pdf_file(Path(pdf_bytes_or_path))
    else:
        return ocr.extract_text_from_pdf_bytes(pdf_bytes_or_path)


if __name__ == "__main__":
    # Test OCR extraction
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists():
            text = extract_text_with_ocr(pdf_path)
            if text:
                print(f"Extracted {len(text)} characters")
                print(text[:1000])
            else:
                print("No text extracted")
        else:
            print(f"File not found: {pdf_path}")
    else:
        print("Usage: python ocr_processor.py <pdf_file>")
