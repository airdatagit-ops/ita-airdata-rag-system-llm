"""
DECEA Publications Scraper Module for Aviation RAG System.

Based on ICAExtractor.py approach - uses Selenium to render JavaScript
and extract PDF links from publication pages.

Usage:
    from parsers.decea_scraper import DECEAScraper

    scraper = DECEAScraper()
    documents = scraper.search(doc_types=["ICA"], limit=100)
"""

import re
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin
from pathlib import Path

import requests
from loguru import logger
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from config import config

# Directory for storing original documents
ORIGINALS_DIR = Path(__file__).parent.parent / "data" / "originals"
DATA_DIR = Path(__file__).parent.parent / "data" / "decea"


def get_doc_type_folder(doc_type: str) -> str:
    """Map document type to folder name."""
    doc_type_lower = (doc_type or "").lower()
    
    type_mapping = {
        "ica": "ica",
        "aca": "ica",
        "mca": "ica",
        "pca": "ica",
        "dca": "ica",
        "tca": "ica",
        "circea": "ica",
        "nsca": "ica",
    }
    
    for key, folder in type_mapping.items():
        if key in doc_type_lower:
            return folder
    
    return "outros"


class DECEAScraper:
    """Scraper for DECEA Publications Portal (publicacoes.decea.mil.br)."""

    BASE_URL = "https://publicacoes.decea.mil.br"
    PUBLICATION_URL = f"{BASE_URL}/publicacao"
    INDEX_URL = f"{BASE_URL}/publicacao/indice"

    def __init__(self, headless: bool = True):
        """
        Initialize DECEA scraper.

        Args:
            headless: Run browser in headless mode (default: True)
        """
        self.headless = headless
        self.driver = None
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        })
        
        # Create data directories
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("DECEAScraper initialized")

    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        if self.driver is not None:
            return
        
        logger.info("Initializing Chrome WebDriver...")
        
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("Chrome WebDriver initialized")

    def _close_driver(self):
        """Close Selenium WebDriver."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

    def __del__(self):
        """Cleanup on destruction."""
        self._close_driver()

    def get_active_icas_from_index(self) -> List[Dict]:
        """
        Fetch all active ICAs from the index page.
        
        Based on ICAExtractor.create_ica_dataframe() method.
        Uses the table at /publicacao/indice to get all active ICAs.
        
        Returns:
            List of ICA metadata dictionaries with numero, titulo, vigor_inicial, origem
        """
        logger.info(f"Fetching ICAs from index: {self.INDEX_URL}")
        
        # Use Selenium to get the page (JavaScript rendered)
        self._init_driver()
        
        try:
            self.driver.get(self.INDEX_URL)
            
            # Wait for table to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            logger.info("Index page loaded")
            
            # Give it a moment for all tables to render
            time.sleep(3)
            
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all tables and look for the ICA table
            # The ICA table has headers with "ICA" in the last column
            tables = soup.find_all('table')
            logger.debug(f"Found {len(tables)} tables")
            
            ica_table = None
            for table in tables:
                headers = table.find_all('th')
                if headers:
                    # Check if any header contains "ICA" (but not other types)
                    header_texts = [h.get_text(strip=True) for h in headers]
                    # The 5th header typically contains the first document number like "ICA63-47"
                    if len(header_texts) >= 5:
                        last_header = header_texts[4] if len(header_texts) > 4 else ""
                        if last_header.startswith('ICA') and not last_header.startswith('IECEA'):
                            ica_table = table
                            logger.info(f"Found ICA table with header: {last_header}")
                            break
            
            if not ica_table:
                logger.error("Could not find ICA table in index page")
                return []
            
            tbody = ica_table.find('tbody')
            if not tbody:
                tbody = ica_table  # Use table directly if no tbody
            
            icas = []
            for tr in tbody.find_all('tr'):
                cells = tr.find_all(['td', 'th'])
                if len(cells) >= 4:
                    numero = cells[0].get_text(strip=True)
                    titulo = cells[1].get_text(strip=True)
                    vigor_inicial = cells[2].get_text(strip=True)
                    origem = cells[3].get_text(strip=True)
                    
                    # Skip header row and empty rows
                    if numero and numero != 'NÚMERO' and numero != 'Número':
                        # Fix the slug format (numero is like "ICA63-47", need "ICA-63-47")
                        if numero.startswith('ICA'):
                            slug = numero.replace('ICA', 'ICA-', 1)
                        else:
                            slug = f"ICA-{numero}"
                        
                        icas.append({
                            'numero': numero,
                            'titulo': titulo,
                            'vigor_inicial': vigor_inicial,
                            'origem': origem,
                            'slug': slug.replace(' ', '-'),
                        })
            
            logger.success(f"Found {len(icas)} ICAs in index")
            return icas
            
        except Exception as e:
            logger.error(f"Error fetching index: {e}")
            return []

    def get_pdf_link_from_page(self, slug: str) -> Optional[str]:
        """
        Use Selenium to get the PDF download link from the publication page.
        
        Based on ICAExtractor.download_ica_documents() method.
        
        Args:
            slug: Document slug (e.g., "ICA-63-47")
            
        Returns:
            PDF download URL or None
        """
        url = f"{self.PUBLICATION_URL}/{slug}"
        
        self._init_driver()
        
        try:
            logger.debug(f"Loading page: {url}")
            self.driver.get(url)
            
            # Wait for table to load (like in ICAExtractor)
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                logger.debug("Table loaded")
            except Exception as e:
                logger.warning(f"Table not found: {e}")
                return None
            
            html_content = self.driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the table with document versions
            # Based on ICAExtractor selector
            table = soup.select_one(
                "div:nth-of-type(3) > div > div:nth-of-type(1) > div > div:nth-of-type(5) > div > table"
            )
            
            if not table:
                # Fallback: find any table with PDF links
                table = soup.find('table')
            
            if not table:
                logger.warning(f"No table found for {slug}")
                return None
            
            # Look for PDF links in the table rows
            for tr in table.find_all('tr'):
                for td in tr.find_all('td'):
                    link = td.find('a')
                    if link and link.get('href'):
                        href = link.get('href')
                        logger.debug(f"Found link: {href}")
                        return href
            
            logger.warning(f"No PDF link found for {slug}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting PDF link: {e}")
            return None

    def download_pdf(self, url: str, slug: str) -> Optional[bytes]:
        """
        Download PDF file from URL.
        
        Args:
            url: PDF URL
            slug: Document slug for logging
            
        Returns:
            PDF content as bytes or None
        """
        try:
            logger.debug(f"Downloading PDF for {slug}: {url[:80]}...")
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Verify it's actually a PDF
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower() or response.content[:4] == b'%PDF':
                    logger.success(f"Downloaded PDF for {slug}: {len(response.content)} bytes")
                    return response.content
                else:
                    logger.warning(f"Got non-PDF content for {slug}: {content_type}")
                    return None
            else:
                logger.warning(f"Failed to download PDF: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None

    def search(
        self,
        doc_types: List[str] = None,
        keywords: List[str] = None,
        limit: int = 100,
        slugs: List[str] = None
    ) -> List[Dict]:
        """
        Search for documents in DECEA portal.

        Args:
            doc_types: Document types to search (e.g., ["ICA", "MCA"])
            keywords: Keywords to filter by (in title)
            limit: Maximum number of documents
            slugs: Specific document slugs to fetch (e.g., ["ICA-63-47"])

        Returns:
            List of document metadata dictionaries
        """
        doc_types = doc_types or ["ICA"]
        
        logger.info(f"Searching DECEA Portal: types={doc_types}, limit={limit}")

        documents = []
        
        if slugs:
            # Fetch specific documents by slug
            for slug in slugs[:limit]:
                doc = self._fetch_document_by_slug(slug)
                if doc:
                    documents.append(doc)
        else:
            # Get all active ICAs from index
            if "ICA" in [t.upper() for t in doc_types]:
                icas = self.get_active_icas_from_index()
                
                for ica in icas[:limit]:
                    # Apply keyword filter if specified
                    if keywords:
                        title_lower = ica.get('titulo', '').lower()
                        if not any(kw.lower() in title_lower for kw in keywords):
                            continue
                    
                    doc = {
                        'slug': ica['slug'],
                        'type': 'ICA',
                        'number': ica['numero'],
                        'title': ica['titulo'],
                        'description': ica['titulo'],
                        'date_published': ica['vigor_inicial'],
                        'origin': ica['origem'],
                        'status': 'PUBLISHED',
                        'doc_type': 'ica',
                        'source_url': f"{self.PUBLICATION_URL}/{ica['slug']}",
                        'pdf_link': None,
                    }
                    documents.append(doc)
                    
                    if len(documents) >= limit:
                        break

        return documents[:limit]

    def _fetch_document_by_slug(self, slug: str) -> Optional[Dict]:
        """
        Fetch document metadata by slug.
        """
        # Parse slug to get type and number
        parts = slug.split('-', 1)
        doc_type = parts[0] if len(parts) > 0 else 'ICA'
        number = parts[1] if len(parts) > 1 else ''
        
        # Get page with Selenium to extract title
        url = f"{self.PUBLICATION_URL}/{slug}"
        
        self._init_driver()
        
        try:
            self.driver.get(url)
            time.sleep(2)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Check if page exists
            if "não conseguimos encontrar" in self.driver.page_source.lower():
                return None
            
            # Get title
            title_tag = soup.find('title')
            title = title_tag.text.replace(' | Publicações DECEA', '').strip() if title_tag else slug
            
            return {
                'slug': slug,
                'type': doc_type,
                'number': number,
                'title': title,
                'description': title,
                'date_published': '',
                'status': 'PUBLISHED',
                'doc_type': 'ica',
                'source_url': url,
                'pdf_link': None,
            }
            
        except Exception as e:
            logger.error(f"Error fetching {slug}: {e}")
            return None

    def get_document_text(self, doc: Dict, save_original: bool = True) -> Optional[str]:
        """
        Fetch document full text content (from PDF).

        Args:
            doc: Document dictionary
            save_original: Whether to save the original PDF

        Returns:
            Document full text content or None
        """
        slug = doc.get('slug', '')
        
        # Get PDF link using Selenium
        pdf_link = self.get_pdf_link_from_page(slug)
        
        if not pdf_link:
            logger.warning(f"No PDF link for: {slug}")
            return doc.get('description', None)
        
        doc['pdf_link'] = pdf_link
        
        # Download PDF
        pdf_content = self.download_pdf(pdf_link, slug)
        
        if not pdf_content:
            return doc.get('description', None)
        
        # Save original PDF if requested
        if save_original:
            self._save_original_document(doc, pdf_content, pdf_link)
        
        # Extract text from PDF
        text = self._extract_pdf_text(pdf_content)
        
        if text and len(text) > 100:
            return text
        else:
            logger.warning(f"Could not extract text from PDF: {slug}")
            return doc.get('description', None)

    def _extract_pdf_text(self, pdf_content: bytes, use_ocr: bool = True) -> Optional[str]:
        """
        Extract text from PDF content.
        
        First tries native text extraction (PyMuPDF, pdfplumber).
        If that fails and use_ocr=True, falls back to OCR for scanned documents.
        
        Args:
            pdf_content: PDF file bytes
            use_ocr: Whether to use OCR for scanned PDFs (default: True)
            
        Returns:
            Extracted text or None
        """
        text = None
        
        try:
            # Try PyMuPDF first (faster)
            try:
                import fitz  # PyMuPDF
                
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                text_parts = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text_parts.append(page.get_text())
                
                doc.close()
                text = '\n'.join(text_parts)
                
                if len(text.strip()) > 100:
                    logger.debug(f"Extracted {len(text)} chars using PyMuPDF")
                    return text
                    
            except ImportError:
                pass
            
            # Fallback to pdfplumber
            try:
                import pdfplumber
                import io
                
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    
                    text = '\n'.join(text_parts)
                    if len(text.strip()) > 100:
                        logger.debug(f"Extracted {len(text)} chars using pdfplumber")
                        return text
                        
            except ImportError:
                logger.warning("No PDF library available (install PyMuPDF or pdfplumber)")
                
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
        
        # If native extraction failed and OCR is enabled, try OCR
        if use_ocr and (text is None or len(text.strip()) < 100):
            logger.info("Native text extraction failed, trying OCR...")
            text = self._extract_with_ocr(pdf_content)
            if text and len(text.strip()) > 100:
                return text
        
        return None
    
    def _extract_with_ocr(self, pdf_content: bytes) -> Optional[str]:
        """
        Extract text from scanned PDF using PaddleOCR.
        
        Args:
            pdf_content: PDF file bytes
            
        Returns:
            Extracted text or None
        """
        try:
            from parsers.ocr_processor import OCRProcessor
            
            ocr = OCRProcessor(use_gpu=False)
            text = ocr.extract_text_from_pdf_bytes(pdf_content, resolution=300)
            
            if text and len(text.strip()) > 50:
                logger.success(f"Extracted {len(text)} chars using OCR")
                return text
            else:
                logger.warning("OCR extraction returned insufficient text")
                return None
                
        except ImportError:
            logger.warning("OCR not available. Install with: pip install paddleocr paddlepaddle")
            return None
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def _save_original_document(self, doc: Dict, content: bytes, url: str):
        """Save original PDF document."""
        try:
            doc_type = doc.get('type', 'ICA').lower()
            folder = ORIGINALS_DIR / get_doc_type_folder(doc_type)
            folder.mkdir(parents=True, exist_ok=True)
            
            slug = doc.get('slug', 'unknown')
            file_path = folder / f"{slug}.pdf"
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.debug(f"Saved PDF: {file_path}")
            
            # Save metadata
            meta_path = folder / f"{slug}.json"
            meta = {
                'slug': slug,
                'type': doc.get('type'),
                'title': doc.get('title'),
                'source_url': doc.get('source_url'),
                'pdf_url': url,
                'downloaded_at': datetime.now().isoformat(),
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving document: {e}")

    def save_document_json(self, doc: Dict, text: str) -> Path:
        """
        Save document as JSON for ingestion.
        
        Args:
            doc: Document metadata
            text: Document text content
            
        Returns:
            Path to saved JSON file
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        slug = doc.get('slug', 'unknown')
        file_path = DATA_DIR / f"{slug}.json"
        
        output = {
            'id': hashlib.md5(slug.encode()).hexdigest(),
            'slug': slug,
            'type': doc.get('type', 'ICA'),
            'number': doc.get('number', ''),
            'title': doc.get('title', ''),
            'description': doc.get('description', ''),
            'date_published': doc.get('date_published', ''),
            'source_url': doc.get('source_url', ''),
            'pdf_link': doc.get('pdf_link', ''),
            'content': text,
            'doc_type': 'ica',
            'scraped_at': datetime.now().isoformat(),
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved JSON: {file_path}")
        return file_path
