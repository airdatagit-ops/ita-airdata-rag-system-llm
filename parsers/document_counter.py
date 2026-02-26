"""
Document Counter Utility for Aviation RAG System.

Counts documents by type in the data directories.
"""

import json
from pathlib import Path
from typing import Dict
from loguru import logger


# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
ORIGINALS_DIR = DATA_DIR / "originals"
LEXML_DIR = DATA_DIR / "lexml"
DECEA_DIR = DATA_DIR / "decea"


def count_documents_by_type() -> Dict[str, Dict]:
    """
    Count documents by type in the data directories.
    Uses lexml and decea JSON files as sources.
    
    Returns:
        Dictionary with counts per document type and totals
    """
    counts = {
        "leis": {"count": 0, "label": "Leis e Decretos"},
        "medidas_provisorias": {"count": 0, "label": "Medidas Provisórias"},
        "acordaos": {"count": 0, "label": "Acórdãos e Jurisprudência"},
        "projetos_lei": {"count": 0, "label": "Projetos de Lei"},
        "ica": {"count": 0, "label": "Instruções de Comando (ICA)"},
        "mca": {"count": 0, "label": "Manual do Comando da Aeronáutica (MCA)"},
        "rbac": {"count": 0, "label": "RBAC e Regulamentos"},
        "portarias": {"count": 0, "label": "Portarias"},
        "resolucoes": {"count": 0, "label": "Resoluções"},
        "outros": {"count": 0, "label": "Outros Documentos"},
    }
    
    total_processed = 0
    
    # Count from processed JSON documents in lexml folder
    lexml_count = 0
    if LEXML_DIR.exists():
        json_files = list(LEXML_DIR.glob("*.json"))
        lexml_count = len(json_files)
        
        # Categorize from lexml JSONs
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                doc_type = (doc.get('doc_type') or 'outros').lower()
                
                # Map to our categories
                if 'lei' in doc_type and 'projeto' not in doc_type:
                    counts["leis"]["count"] += 1
                elif 'decreto' in doc_type:
                    counts["leis"]["count"] += 1
                elif 'medida.provisoria' in doc_type or 'mpv' in doc_type:
                    counts["medidas_provisorias"]["count"] += 1
                elif 'acordao' in doc_type or 'jurisprudencia' in doc_type:
                    counts["acordaos"]["count"] += 1
                elif 'projeto' in doc_type:
                    counts["projetos_lei"]["count"] += 1
                elif 'rbac' in doc_type or 'regulamento' in doc_type:
                    counts["rbac"]["count"] += 1
                elif 'portaria' in doc_type:
                    counts["portarias"]["count"] += 1
                elif 'resolucao' in doc_type or 'resolução' in doc_type:
                    counts["resolucoes"]["count"] += 1
                else:
                    counts["outros"]["count"] += 1
            except Exception as e:
                logger.warning(f"Error reading {json_path}: {e}")
                counts["outros"]["count"] += 1
    
    total_processed += lexml_count
    
    # Count from DECEA folder (ICAs, MCAs, etc.)
    decea_count = 0
    if DECEA_DIR.exists():
        json_files = list(DECEA_DIR.glob("*.json"))
        decea_count = len(json_files)
        
        # Categorize DECEA documents
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                doc_type = (doc.get('type') or doc.get('doc_type') or '').upper()
                slug = (doc.get('slug') or '').upper()
                
                # Map DECEA document types
                if doc_type == 'ICA' or slug.startswith('ICA'):
                    counts["ica"]["count"] += 1
                elif doc_type == 'MCA' or slug.startswith('MCA'):
                    counts["mca"]["count"] += 1
                elif doc_type in ['PCA', 'DCA', 'TCA', 'NSCA', 'CIRCEA']:
                    # Other DECEA document types go to ICA category for now
                    counts["ica"]["count"] += 1
                else:
                    counts["outros"]["count"] += 1
            except Exception as e:
                logger.warning(f"Error reading DECEA file {json_path}: {e}")
                counts["outros"]["count"] += 1
    
    total_processed += decea_count
    
    # Count original files (HTML from LexML, PDFs from DECEA)
    total_originals = 0
    if ORIGINALS_DIR.exists():
        for folder in ORIGINALS_DIR.iterdir():
            if folder.is_dir():
                # Count HTML files
                total_originals += len(list(folder.glob("*.html")))
                # Count PDF files (DECEA originals)
                total_originals += len(list(folder.glob("*.pdf")))
    
    return {
        "by_type": counts,
        "total_originals": total_originals,
        "total_processed": total_processed,
        "sources": {
            "lexml": lexml_count,
            "decea": decea_count
        }
    }


def get_document_stats() -> Dict:
    """
    Get comprehensive document statistics.
    
    Returns:
        Dictionary with all document statistics
    """
    counts = count_documents_by_type()
    
    return {
        "documents": counts,
        "summary": {
            "total_documents": counts["total_originals"],
            "total_processed": counts["total_processed"],
            "categories": len([c for c in counts["by_type"].values() if c["count"] > 0]),
            "sources": counts.get("sources", {})
        }
    }


if __name__ == "__main__":
    # Test the counter
    import json as json_module
    stats = get_document_stats()
    print(json_module.dumps(stats, indent=2, ensure_ascii=False))
