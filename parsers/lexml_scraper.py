"""
LexML Scraper Module for Aviation RAG System.

This module scrapes legal documents from LexML Brasil web portal.
Uses web scraping since the SRU API is no longer available.

Usage:
    from parsers.lexml_scraper import LexMLScraper

    scraper = LexMLScraper()
    documents = scraper.search(keywords=["aviação", "ANAC"], limit=100)
"""

import re
import time
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin, urlencode
from pathlib import Path

import requests
from loguru import logger
from bs4 import BeautifulSoup

from config import config
from parsers.document_tracker import DocumentTracker, get_tracker

# Directory for storing original documents
ORIGINALS_DIR = Path(__file__).parent.parent / "data" / "originals"


def get_doc_type_folder(doc_type: str) -> str:
    """Map document type to folder name."""
    doc_type_lower = (doc_type or "").lower()
    
    type_mapping = {
        "lei": "leis",
        "lei.complementar": "leis",
        "lei.ordinaria": "leis",
        "decreto": "leis",
        "decreto.lei": "leis",
        "ica": "ica",
        "instrucao": "ica",
        "rbac": "rbac",
        "regulamento": "rbac",
        "portaria": "portarias",
        "resolucao": "resolucoes",
        "resolução": "resolucoes",
    }
    
    for key, folder in type_mapping.items():
        if key in doc_type_lower:
            return folder
    
    return "outros"


class LexMLScraper:
    """Scraper for LexML Brasil Web Portal (web scraping approach)."""

    # Base URLs for LexML portal
    SEARCH_URL = "https://www.lexml.gov.br/busca/search"
    BASE_URL = "https://www.lexml.gov.br"
    NORMAS_URL = "https://normas.leg.br"

    def __init__(self, max_records_per_page: int = None, skip_duplicates: bool = True):
        """
        Initialize LexML scraper.

        Args:
            max_records_per_page: Maximum records per page (default: 20)
            skip_duplicates: Whether to skip already downloaded documents (default: True)
        """
        self.max_records_per_page = max_records_per_page or config.LEXML_MAX_RECORDS_PER_PAGE or 20
        self.skip_duplicates = skip_duplicates
        self.tracker = get_tracker() if skip_duplicates else None
        self.session = requests.Session()
        # Set headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        logger.info(f"LexMLScraper initialized (Web Scraping Mode, skip_duplicates={skip_duplicates})")

    def is_duplicate(self, doc: Dict, content: str = None) -> bool:
        """
        Check if a document is a duplicate (already downloaded).
        
        Args:
            doc: Document metadata dictionary
            content: Optional document content for hash verification
            
        Returns:
            True if document was already downloaded
        """
        if not self.skip_duplicates or not self.tracker:
            return False
        return self.tracker.is_duplicate(doc, content)
    
    def register_document(self, doc: Dict, content: str = None, file_path: str = None):
        """
        Register a downloaded document in the tracker.
        
        Args:
            doc: Document metadata dictionary
            content: Document content
            file_path: Path where document was saved
        """
        if self.tracker:
            self.tracker.register_document(doc, content, file_path, source="lexml")

    def get_tracker_stats(self) -> Dict:
        """Get document tracker statistics."""
        if self.tracker:
            return self.tracker.get_stats()
        return {}

    def search(
        self,
        keywords: List[str] = None,
        doc_type: str = None,
        authority_level: str = "federal",
        limit: int = 100,
        start_record: int = 0
    ) -> List[Dict]:
        """
        Search for documents in LexML web portal.

        Args:
            keywords: Keywords to search (e.g., ["aviação", "ANAC"])
            doc_types: Document types (e.g., ["lei", "decreto"])
            authority_level: Authority level (federal, estadual, municipal)
            limit: Maximum number of documents
            start_record: Starting record offset

        Returns:
            List of document metadata dictionaries
        """
        keywords = keywords or config.lexml_keywords_list
        query = self._build_search_query(keywords, doc_type, authority_level)

        logger.info(f"Searching LexML Portal: {query} (limit={limit})")

        documents = []
        current_offset = start_record
        page_size = 20  # LexML portal returns 20 per page

        while len(documents) < limit:
            try:
                # Build search URL with pagination
                # LexML uses 'keyword' param and 'startDoc' for pagination
                params = query
                if current_offset > 0:
                    params += f";startDoc={current_offset + 1}"
                
                search_url = f"{self.SEARCH_URL}?{params}"
                logger.debug(f"Fetching: {search_url}")
                
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code != 200:
                    logger.warning(f"Search returned status {response.status_code}")
                    break
                
                batch_docs = self._parse_search_results(response.text)

                if not batch_docs:
                    logger.info("No more documents found")
                    break

                documents.extend(batch_docs)
                logger.info(f"Retrieved {len(documents)}/{limit} documents")

                current_offset += len(batch_docs)
                time.sleep(1.0)  # Rate limiting - be respectful

            except Exception as e:
                logger.error(f"Error fetching documents: {e}")
                break

        return documents[:limit]

    def _build_search_query(
        self,
        keywords: List[str],
        doc_type: str = None,
        authority_level: str = "federal"
    ) -> str:
        """Build search query string for the web portal."""
        # LexML uses simple keyword search
        # Join keywords with spaces (AND search)
        query_parts = []
        
        if keywords:
            keywords_str = "keyword=" + "+".join(keywords)
            query_parts.append(keywords_str)
        
        if doc_type:
            docs_str = f";f1-tipoDocumento={doc_type}"
            query_parts.append(docs_str)
        
        search_query = ' '.join(query_parts) if query_parts else 'lei federal'
        logger.info('Search query constructed: {}'.format(search_query))
        return search_query

    def _parse_search_results(self, html_content: str) -> List[Dict]:
        """Parse search results from HTML page."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            documents = []

            # Find result items with class 'docHit'
            result_items = soup.find_all('div', class_='docHit')
            
            logger.debug(f"Found {len(result_items)} docHit elements")

            for item in result_items:
                doc = self._parse_result_item(item)
                if doc:
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return []

    def _parse_result_item(self, item) -> Optional[Dict]:
        """Parse individual search result item (docHit)."""
        try:
            doc = {}
            
            # Find all table rows
            rows = item.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    label_elem = cols[1]
                    value_elem = cols[2]
                    
                    label = label_elem.get_text(strip=True).replace(':', '').lower()
                    
                    # Handle different fields
                    if 'título' in label or 'titulo' in label:
                        link = value_elem.find('a')
                        if link:
                            doc['title'] = link.get_text(strip=True)
                            href = link.get('href', '')
                            if href:
                                doc['url'] = urljoin(self.BASE_URL, href)
                                # Extract URN from URL if present
                                if '/urn/' in href:
                                    urn_part = href.split('/urn/')[-1]
                                    doc['urn'] = urn_part
                    elif 'urn' in label:
                        # Get text and clean up any span tags
                        urn_text = value_elem.get_text(strip=True)
                        # Remove any extra whitespace
                        urn_text = ' '.join(urn_text.split())
                        if urn_text.startswith('urn:lex:'):
                            doc['urn'] = urn_text
                    elif 'data' in label:
                        doc['date'] = value_elem.get_text(strip=True)
                    elif 'ementa' in label:
                        # Clean up ementa text (remove highlight spans)
                        ementa_text = value_elem.get_text(strip=True)
                        doc['description'] = ementa_text[:500]
                    elif 'autoridade' in label:
                        doc['authority'] = value_elem.get_text(strip=True)
                    elif 'localidade' in label:
                        doc['location'] = value_elem.get_text(strip=True)
                    elif 'assuntos' in label:
                        doc['subjects'] = value_elem.get_text(strip=True)

            # Parse URN for additional metadata
            if doc.get('urn'):
                doc.update(self._parse_urn(doc['urn']))

            # Only return if we have at least a title or URN
            if doc.get('title') or doc.get('urn'):
                return doc

            return None

        except Exception as e:
            logger.warning(f"Error parsing result item: {e}")
            return None

    def _parse_urn(self, urn: str) -> Dict:
        """
        Parse LexML URN.

        Example: urn:lex:br:federal:lei:1993-06-21;8666
        """
        parts = urn.split(':')
        if len(parts) >= 6:
            date_and_number = parts[5].split(';')
            return {
                "authority": parts[3] if len(parts) > 3 else None,
                "doc_type": parts[4] if len(parts) > 4 else None,
                "publication_date": date_and_number[0] if date_and_number else None,
                "number": date_and_number[1] if len(date_and_number) > 1 else None
            }
        return {}

    def download_document(self, doc: Dict, output_path: str, save_original: bool = True) -> bool:
        """
        Download full content for a document.

        Args:
            doc: Document dictionary with 'url' or 'urn'
            output_path: Path to save content
            save_original: Whether to save the original HTML

        Returns:
            True if successful
        """
        url = doc.get('url')
        urn = doc.get('urn')
        
        if not url and urn:
            # Try to construct URL from URN
            url = f"{self.NORMAS_URL}/?urn={quote(urn)}"
        
        if not url:
            logger.warning(f"No URL available for document: {doc.get('title', 'unknown')}")
            return False

        try:
            logger.debug(f"Downloading from: {url}")
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Failed to download (status {response.status_code}): {url}")
                return False

            # Save original HTML if requested
            if save_original:
                self._save_original_document(doc, response.text, url)

            # Parse the document page to extract content
            content = self._extract_document_content(response.text, doc)
            
            if content:
                # Save as JSON with metadata and content
                output_data = {
                    **doc,
                    'content': content,
                    'source_url': url
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Downloaded document: {output_path}")
                return True
            else:
                logger.warning(f"Could not extract content from: {url}")
                return False

        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return False

    def _save_original_document(self, doc: Dict, html_content: str, source_url: str) -> bool:
        """
        Save the original HTML document to the originals folder.
        
        Args:
            doc: Document metadata
            html_content: Raw HTML content
            source_url: URL the document was downloaded from
            
        Returns:
            True if saved successfully
        """
        try:
            # Determine folder based on document type
            doc_type = doc.get('doc_type', 'outros')
            folder = get_doc_type_folder(doc_type)
            
            # Create folder if needed
            folder_path = ORIGINALS_DIR / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from document info
            doc_number = doc.get('number', '')
            doc_date = doc.get('publication_date', '')
            title = doc.get('title', 'documento')
            
            # Clean title for filename
            clean_title = re.sub(r'[^\w\s\-]', '', title)[:50].strip().replace(' ', '_')
            
            # Create unique filename
            if doc_number and doc_date:
                filename = f"{doc_type}_{doc_number}_{doc_date}"
            else:
                # Use hash of URL for unique identifier
                url_hash = hashlib.md5(source_url.encode()).hexdigest()[:8]
                filename = f"{clean_title}_{url_hash}"
            
            # Save HTML file
            html_path = folder_path / f"{filename}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save metadata JSON alongside
            meta_path = folder_path / f"{filename}_meta.json"
            metadata = {
                **doc,
                'source_url': source_url,
                'original_file': str(html_path),
                'downloaded_at': datetime.now().isoformat()
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved original document: {html_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving original document: {e}")
            return False

    def _extract_document_content(self, html_content: str, doc: Dict) -> Optional[str]:
        """Extract main content from document page."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()

            # Try various selectors for content
            content_selectors = [
                '.texto-norma',
                '.conteudo-norma', 
                '.document-content',
                '#texto',
                '.texto',
                'article',
                '.content',
                'main',
                '#conteudo'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    text = content_elem.get_text(separator='\n', strip=True)
                    if len(text) > 100:  # Reasonable content length
                        return text

            # Fallback: get body text
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
                # Clean up excessive whitespace
                text = re.sub(r'\n{3,}', '\n\n', text)
                if len(text) > 100:
                    return text

            return None

        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return None

    def download_xml(self, urn: str, output_path: str) -> bool:
        """
        Download document content (backward compatibility).
        Now saves as JSON since XML API is not available.

        Args:
            urn: Document URN or document dict
            output_path: Path to save content

        Returns:
            True if successful
        """
        # Handle both URN string and document dict
        if isinstance(urn, str):
            doc = {'urn': urn}
        else:
            doc = urn
            
        # Change extension to .json if .xml
        if output_path.endswith('.xml'):
            output_path = output_path.replace('.xml', '.json')
            
        return self.download_document(doc, output_path)

    def get_document_text(self, doc: Dict, save_original: bool = True) -> Optional[str]:
        """
        Fetch document full text content by following links to source sites.

        Strategy:
        1. Access LexML page to find links to Senado/Planalto/Câmara
        2. Follow link to get full law text
        3. Extract text from the source page
        4. Save original HTML if requested

        Args:
            doc: Document dictionary
            save_original: Whether to save the original HTML

        Returns:
            Document full text content or None
        """
        url = doc.get('url')
        urn = doc.get('urn')
        
        if not url and urn:
            url = f"{self.BASE_URL}/urn/{quote(urn)}"
        
        if not url:
            return None

        try:
            # Step 1: Access LexML page to find source links
            logger.debug(f"Accessing LexML page: {url}")
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to access LexML page: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Step 2: Find link to full text source (prioritize Senado, then Planalto)
            source_url = None
            
            # Look for Senado link first (has structured text)
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                if 'legis.senado.leg.br/norma' in href:
                    source_url = href
                    logger.debug(f"Found Senado link: {source_url}")
                    break
            
            # Fallback to Planalto
            if not source_url:
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '')
                    if 'planalto.gov.br' in href and 'legislacao' in href:
                        source_url = href
                        logger.debug(f"Found Planalto link: {source_url}")
                        break
            
            if not source_url:
                logger.warning(f"No source link found for: {doc.get('title', url)}")
                # Fallback: return ementa/description if available
                return doc.get('description', None)
            
            # Step 3: Fetch full text from source (and save original)
            text = self._fetch_full_text_from_source(source_url, doc, save_original)
            
            if text:
                return text
            else:
                # Fallback to ementa
                return doc.get('description', None)
                
        except Exception as e:
            logger.error(f"Error fetching document text: {e}")
            return doc.get('description', None)

    def _fetch_full_text_from_source(self, source_url: str, doc: Dict = None, save_original: bool = True) -> Optional[str]:
        """
        Fetch full text from Senado or Planalto source.
        
        Args:
            source_url: URL to the source site (Senado/Planalto)
            doc: Document metadata for saving original
            save_original: Whether to save the original HTML
            
        Returns:
            Full text of the law or None
        """
        try:
            # Access source page
            response = self.session.get(source_url, timeout=30)
            if response.status_code != 200:
                logger.warning(f"Failed to access source: {response.status_code}")
                return None
            
            # Save original if requested
            if save_original and doc:
                self._save_original_document(doc, response.text, source_url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # For Senado: need to find link to publication text
            if 'senado.leg.br' in source_url:
                return self._extract_senado_text(soup, source_url, doc, save_original)
            elif 'planalto.gov.br' in source_url:
                return self._extract_planalto_text(soup)
            else:
                return self._extract_document_content(response.text, {})
                
        except Exception as e:
            logger.error(f"Error fetching from source: {e}")
            return None

    def _extract_senado_text(self, soup: BeautifulSoup, base_url: str, doc: Dict = None, save_original: bool = True) -> Optional[str]:
        """
        Extract full text from Senado Legis page.
        
        The Senado page has links like "Ver texto no Sigen" that lead to the full text.
        """
        try:
            # Find publication links
            pub_links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                text = a.get_text(strip=True).lower()
                if 'publicacao' in href or 'sigen' in text.lower():
                    # Build full URL
                    if href.startswith('/'):
                        full_url = f"https://legis.senado.leg.br{href}"
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        # Relative URL - combine with base
                        base_path = '/'.join(base_url.split('/')[:-1])
                        full_url = f"{base_path}/{href}"
                    pub_links.append(full_url)
            
            if not pub_links:
                logger.warning("No publication links found on Senado page")
                return None
            
            # Try first publication link (usually the original)
            pub_url = pub_links[0]
            logger.debug(f"Fetching Senado publication: {pub_url}")
            
            response = self.session.get(pub_url, timeout=30)
            if response.status_code != 200:
                return None
            
            # Save the publication page as original (this is the full law text)
            if save_original and doc:
                self._save_original_document(doc, response.text, pub_url)
            
            pub_soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from #conteudoPrincipal
            content = pub_soup.select_one('#conteudoPrincipal')
            if content:
                # Remove navigation elements
                for nav in content.find_all(['nav', 'button', 'a']):
                    if nav.get('class') and any('btn' in c for c in nav.get('class', [])):
                        nav.decompose()
                
                text = content.get_text(separator='\n', strip=True)
                
                # Clean up - remove header text before the law title
                if 'LEI' in text or 'DECRETO' in text or 'RESOLUÇÃO' in text:
                    # Find where the actual law text starts
                    for marker in ['LEI ', 'DECRETO ', 'RESOLUÇÃO ', 'PORTARIA ']:
                        if marker in text:
                            idx = text.find(marker)
                            if idx > 0 and idx < 500:  # Header shouldn't be too long
                                text = text[idx:]
                            break
                
                if len(text) > 200:  # Meaningful content
                    logger.info(f"Extracted {len(text)} chars from Senado")
                    return text
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting Senado text: {e}")
            return None

    def _extract_planalto_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract full text from Planalto page."""
        try:
            # Planalto usually has text in #textoNorma or similar
            selectors = ['#textoNorma', '.textoNorma', '#conteudo', '.conteudo', 'article']
            
            for sel in selectors:
                elem = soup.select_one(sel)
                if elem:
                    text = elem.get_text(separator='\n', strip=True)
                    if len(text) > 200:
                        logger.info(f"Extracted {len(text)} chars from Planalto")
                        return text
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting Planalto text: {e}")
            return None


if __name__ == "__main__":
    """Example usage."""
    scraper = LexMLScraper()

    # Search for aviation-related laws
    documents = scraper.search(
        keywords=["aviação", "aeronave", "ANAC"],
        doc_types=["lei", "decreto"],
        limit=10
    )

    print(f"\nFound {len(documents)} documents:")
    for doc in documents[:5]:
        print(f"- {doc.get('title', 'N/A')[:60]}...")
        print(f"  URL: {doc.get('url', 'N/A')}")
        print(f"  URN: {doc.get('urn', 'N/A')}")
        print()
