#!/usr/bin/env python3
"""
Script para copiar todos os arquivos PDF da pasta data/originals/sislaer
para ocr_evaluation/test_data
"""
import os
import shutil
from pathlib import Path

def copy_pdfs():
    # Definir caminhos
    source_dir = Path("../data/originals/sislaer")
    dest_dir = Path("./test_data")
    
    # Verificar se pasta origem existe
    if not source_dir.exists():
        print(f"❌ Pasta origem não encontrada: {source_dir.resolve()}")
        return False
    
    # Criar pasta destino se não existir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos os PDFs
    pdf_files = list(source_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ Nenhum arquivo PDF encontrado em", source_dir.resolve())
        return False
    
    print(f"📁 Encontrados {len(pdf_files)} arquivo(s) PDF")
    print(f"📍 Copiando de: {source_dir.resolve()}")
    print(f"📍 Copiando para: {dest_dir.resolve()}\n")
    
    # Copiar cada PDF
    copied_count = 0
    for pdf_file in pdf_files:
        dest_file = dest_dir / pdf_file.name
        try:
            shutil.copy2(pdf_file, dest_file)
            print(f"✅ {pdf_file.name}")
            copied_count += 1
        except Exception as e:
            print(f"❌ Erro ao copiar {pdf_file.name}: {e}")
    
    print(f"\n✨ Concluído! {copied_count}/{len(pdf_files)} arquivos copiados com sucesso")
    return True

if __name__ == "__main__":
    copy_pdfs()
