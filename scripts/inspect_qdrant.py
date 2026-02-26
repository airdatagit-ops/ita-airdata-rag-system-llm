"""
Script para inspecionar a estrutura de dados retornada pelo Qdrant.
"""

import json
from loguru import logger
from database.qdrant_manager import QdrantManager
from models.embeddings import EmbeddingModel
from tabulate import tabulate


def inspect_collection():
    """Inspeciona informações da coleção."""
    logger.info("=" * 60)
    logger.info("📊 INFORMAÇÕES DA COLEÇÃO QDRANT")
    logger.info("=" * 60)
    
    manager = QdrantManager()
    info = manager.get_collection_info()
    
    print(json.dumps(info, indent=2))
    logger.success(f"Total de vetores: {info['vectors_count']}")


def inspect_payload_structure():
    """Mostra a estrutura exata dos campos em cada registro."""
    logger.info("\n" + "=" * 80)
    logger.info("📋 ESTRUTURA DE CAMPOS (COLUNAS) EM CADA REGISTRO")
    logger.info("=" * 80)
    
    manager = QdrantManager()
    embeddings = EmbeddingModel()
    
    # Criar um vetor de consulta
    query_text = "regulations aircraft"
    query_vector = embeddings.encode(query_text)
    
    # Buscar
    results = manager.search(
        query_vector=query_vector,
        limit=1,
        with_payload=True
    )
    
    if not results:
        logger.warning("Nenhum resultado encontrado no banco de dados")
        return
    
    first_result = results[0]
    payload = first_result.payload
    
    logger.success(f"✓ Encontrado 1 registro (ID: {first_result.id})\n")
    
    # Mostrar estrutura
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ CAMPOS PRINCIPAIS DE UM REGISTRO (Payload)                          │")
    print("└─────────────────────────────────────────────────────────────────────┘\n")
    
    fields_data = []
    for field_name, field_value in payload.items():
        field_type = type(field_value).__name__
        
        # Formatar valor para exibição
        if isinstance(field_value, dict):
            value_preview = "{ ... }" if field_value else "{}"
        elif isinstance(field_value, (list, tuple)):
            value_preview = f"[{len(field_value)} itens]"
        elif isinstance(field_value, str):
            value_preview = field_value[:60] + ("..." if len(field_value) > 60 else "")
        else:
            value_preview = str(field_value)[:60]
        
        fields_data.append([field_name, field_type, value_preview])
    
    # Exibir tabela
    headers = ["CAMPO", "TIPO", "EXEMPLO/PREVIEW"]
    print(tabulate(fields_data, headers=headers, tablefmt="grid"))
    
    # Detalhes de cada campo
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│ DETALHES COMPLETOS DE CADA CAMPO                                    │")
    print("└─────────────────────────────────────────────────────────────────────┘\n")
    
    for field_name, field_value in payload.items():
        print(f"📌 {field_name}:")
        print(f"   Tipo: {type(field_value).__name__}")
        
        if isinstance(field_value, dict):
            print(f"   Subcamp0s: {list(field_value.keys())}")
            print(f"   Valor: {json.dumps(field_value, indent=6, ensure_ascii=False)}")
        elif isinstance(field_value, (list, tuple)):
            print(f"   Quantidade de itens: {len(field_value)}")
            print(f"   Primeiros itens: {field_value[:3] if len(field_value) > 3 else field_value}")
        else:
            print(f"   Valor: {field_value}")
        
        print()
    
    # Campos especiais do Qdrant
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│ CAMPOS ESPECIAIS DO QDRANT (Não armazenados no payload)             │")
    print("└─────────────────────────────────────────────────────────────────────┘\n")
    
    special_fields = [
        ["id", "string", f"Identificador único do ponto: {first_result.id}"],
        ["score", "float", f"Similaridade do resultado (0-1): {first_result.score}"],
        ["vector", "list[float]", f"Vetor de embedding (dimensão configurada)"]
    ]
    
    print(tabulate(special_fields, headers=["CAMPO", "TIPO", "DESCRIÇÃO"], tablefmt="grid"))


def compare_multiple_records():
    """Compara vários registros para identificar campos opcionais."""
    logger.info("\n" + "=" * 80)
    logger.info("🔍 COMPARAÇÃO DE MÚLTIPLOS REGISTROS")
    logger.info("=" * 80)
    
    manager = QdrantManager()
    embeddings = EmbeddingModel()
    
    query_vector = embeddings.encode("regulations")
    results = manager.search(query_vector=query_vector, limit=5, with_payload=True)
    
    if not results:
        logger.warning("Nenhum resultado encontrado")
        return
    
    logger.success(f"✓ Analisando {len(results)} registros\n")
    
    # Coletar todos os campos únicos
    all_fields = set()
    for result in results:
        all_fields.update(result.payload.keys())
    
    all_fields = sorted(list(all_fields))
    
    # Tabela de presença de campos
    comparison_data = []
    for field in all_fields:
        presence = [
            "✓" if field in result.payload else "" 
            for result in results
        ]
        comparison_data.append([field] + presence)
    
    headers = ["CAMPO"] + [f"Record {i+1}" for i in range(len(results))]
    print(tabulate(comparison_data, headers=headers, tablefmt="grid"))
    
    print(f"\n📌 Resumo:")
    print(f"   - Total de campos únicos encontrados: {len(all_fields)}")
    obrigatorios = 0
    opcionais = 0
    for field in all_fields:
        count = sum(1 for result in results if field in result.payload)
        if count == len(results):
            obrigatorios += 1
        else:
            opcionais += 1
    
    print(f"   - Campos em TODOS os registros (obrigatórios): {obrigatorios}")
    print(f"   - Campos em ALGUNS registros (opcionais): {opcionais}")


if __name__ == "__main__":
    try:
        inspect_collection()
        inspect_payload_structure()
        compare_multiple_records()
        
        logger.success("\n✅ Inspeção concluída!")
        
    except Exception as e:
        logger.error(f"Erro durante inspeção: {e}")
        import traceback
        traceback.print_exc()
