"""
Document Tracker for Aviation RAG System.

Tracks downloaded documents to avoid re-downloading duplicates.
Maintains a log of processed documents by URN, URL, and content hash.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from loguru import logger


# Default tracker file location
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TRACKER_FILE = DATA_DIR / "document_tracker.json"


class DocumentTracker:
    """
    Tracks downloaded and processed documents to avoid duplicates.
    
    Stores:
    - URNs of downloaded documents
    - URLs of downloaded documents  
    - Content hashes to detect duplicate content
    - Download timestamps
    """
    
    def __init__(self, tracker_file: Path = None):
        """
        Initialize the document tracker.
        
        Args:
            tracker_file: Path to the tracker JSON file
        """
        self.tracker_file = tracker_file or DEFAULT_TRACKER_FILE
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._load_tracker()
        logger.info(f"DocumentTracker initialized with {len(self.data['documents'])} tracked documents")
    
    def _load_tracker(self):
        """Load tracker data from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                # Ensure required keys exist
                self.data.setdefault('documents', {})
                self.data.setdefault('urns', {})
                self.data.setdefault('urls', {})
                self.data.setdefault('content_hashes', {})
            except Exception as e:
                logger.warning(f"Error loading tracker file: {e}. Creating new tracker.")
                self._init_empty_tracker()
        else:
            self._init_empty_tracker()
    
    def _init_empty_tracker(self):
        """Initialize empty tracker data structure."""
        self.data = {
            'documents': {},      # doc_id -> document metadata
            'urns': {},           # urn -> doc_id
            'urls': {},           # url -> doc_id
            'content_hashes': {}, # content_hash -> doc_id
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_tracker(self):
        """Save tracker data to file."""
        try:
            self.data['last_updated'] = datetime.now().isoformat()
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving tracker file: {e}")
    
    def _generate_doc_id(self, doc: Dict) -> str:
        """Generate a unique document ID."""
        # Use URN if available, otherwise URL, otherwise hash of title
        if doc.get('urn'):
            return hashlib.md5(doc['urn'].encode()).hexdigest()
        elif doc.get('url'):
            return hashlib.md5(doc['url'].encode()).hexdigest()
        elif doc.get('title'):
            return hashlib.md5(doc['title'].encode()).hexdigest()
        else:
            # Fallback to random-ish hash
            import uuid
            return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of document content."""
        if not content:
            return None
        # Normalize content before hashing
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def is_duplicate(self, doc: Dict, content: str = None) -> bool:
        """
        Check if a document is a duplicate.
        
        Checks by:
        1. URN (if available)
        2. URL (if available)
        3. Content hash (if content provided)
        
        Args:
            doc: Document metadata dictionary
            content: Optional document content for hash check
            
        Returns:
            True if document is a duplicate
        """
        # Check by URN
        urn = doc.get('urn')
        if urn and urn in self.data['urns']:
            logger.debug(f"Duplicate found by URN: {urn}")
            return True
        
        # Check by URL
        url = doc.get('url')
        if url and url in self.data['urls']:
            logger.debug(f"Duplicate found by URL: {url}")
            return True
        
        # Check by content hash
        if content:
            content_hash = self._get_content_hash(content)
            if content_hash and content_hash in self.data['content_hashes']:
                logger.debug(f"Duplicate found by content hash")
                return True
        
        return False
    
    def get_duplicate_info(self, doc: Dict) -> Optional[Dict]:
        """
        Get information about the existing duplicate document.
        
        Args:
            doc: Document metadata to check
            
        Returns:
            Dictionary with duplicate document info, or None
        """
        urn = doc.get('urn')
        if urn and urn in self.data['urns']:
            doc_id = self.data['urns'][urn]
            return self.data['documents'].get(doc_id)
        
        url = doc.get('url')
        if url and url in self.data['urls']:
            doc_id = self.data['urls'][url]
            return self.data['documents'].get(doc_id)
        
        return None
    
    def register_document(
        self,
        doc: Dict,
        content: str = None,
        file_path: str = None,
        source: str = "unknown"
    ) -> str:
        """
        Register a downloaded document in the tracker.
        
        Args:
            doc: Document metadata dictionary
            content: Document content (for hash)
            file_path: Path where document was saved
            source: Source identifier (lexml, decea, etc.)
            
        Returns:
            Document ID
        """
        doc_id = self._generate_doc_id(doc)
        
        # Store document metadata
        self.data['documents'][doc_id] = {
            'id': doc_id,
            'title': doc.get('title'),
            'urn': doc.get('urn'),
            'url': doc.get('url'),
            'doc_type': doc.get('doc_type') or doc.get('type'),
            'file_path': str(file_path) if file_path else None,
            'source': source,
            'downloaded_at': datetime.now().isoformat()
        }
        
        # Index by URN
        if doc.get('urn'):
            self.data['urns'][doc['urn']] = doc_id
        
        # Index by URL
        if doc.get('url'):
            self.data['urls'][doc['url']] = doc_id
        
        # Index by content hash
        if content:
            content_hash = self._get_content_hash(content)
            if content_hash:
                self.data['content_hashes'][content_hash] = doc_id
                self.data['documents'][doc_id]['content_hash'] = content_hash
        
        self._save_tracker()
        logger.debug(f"Registered document: {doc.get('title', doc_id)[:50]}")
        
        return doc_id
    
    def get_stats(self) -> Dict:
        """Get tracker statistics."""
        return {
            'total_documents': len(self.data['documents']),
            'unique_urns': len(self.data['urns']),
            'unique_urls': len(self.data['urls']),
            'content_hashes': len(self.data['content_hashes']),
            'created_at': self.data.get('created_at'),
            'last_updated': self.data.get('last_updated')
        }
    
    def get_all_urns(self) -> Set[str]:
        """Get all tracked URNs."""
        return set(self.data['urns'].keys())
    
    def get_all_urls(self) -> Set[str]:
        """Get all tracked URLs."""
        return set(self.data['urls'].keys())
    
    def clear(self):
        """Clear all tracker data."""
        self._init_empty_tracker()
        self._save_tracker()
        logger.info("Document tracker cleared")
    
    def remove_document(self, doc_id: str):
        """Remove a document from the tracker."""
        if doc_id not in self.data['documents']:
            return
        
        doc_info = self.data['documents'][doc_id]
        
        # Remove from indexes
        if doc_info.get('urn') and doc_info['urn'] in self.data['urns']:
            del self.data['urns'][doc_info['urn']]
        
        if doc_info.get('url') and doc_info['url'] in self.data['urls']:
            del self.data['urls'][doc_info['url']]
        
        if doc_info.get('content_hash') and doc_info['content_hash'] in self.data['content_hashes']:
            del self.data['content_hashes'][doc_info['content_hash']]
        
        # Remove document
        del self.data['documents'][doc_id]
        
        self._save_tracker()
        logger.debug(f"Removed document: {doc_id}")


# Global tracker instance (lazy loaded)
_tracker_instance = None


def get_tracker() -> DocumentTracker:
    """Get the global document tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DocumentTracker()
    return _tracker_instance


if __name__ == "__main__":
    # Test the tracker
    tracker = DocumentTracker()
    print(f"Tracker stats: {tracker.get_stats()}")
