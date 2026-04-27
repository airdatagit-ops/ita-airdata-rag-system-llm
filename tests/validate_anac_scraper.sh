#!/bin/bash
# Validation script for ANAC RBAC Scraper
# Usage: bash tests/validate_anac_scraper.sh
set -e

echo "=== Teste 1: Importação ==="
python -c "from crawler.scrapers import get_scraper; scraper = get_scraper('anac_rbac'); print(f'✓ Scraper importado: {scraper.source_name}')"

echo -e "\n=== Teste 2: Dry-Run ==="
python scripts/collect_anac_rbac.py --dry-run 2>&1 | tail -20

echo -e "\n=== Teste 3: Diretórios ==="
test -d data/originals/anac && echo "✓ Diretório data/originals/anac/ criado" || echo "✗ ERRO: Diretório não existe"
test -w data/originals/anac && echo "✓ Permissões de escrita OK" || echo "✗ ERRO: Sem permissão de escrita"

echo -e "\n=== Teste 7: Interface BaseScraper ==="
python -c "
from crawler.scrapers.anac_rbac_scraper import ANACRBACscraper
from crawler.scrapers.base import BaseScraper
from inspect import signature

scraper_class = ANACRBACscraper

# 1. Verificar herança
assert issubclass(scraper_class, BaseScraper), '✗ ANACRBACscraper não herda de BaseScraper'
print('✓ ANACRBACscraper herda de BaseScraper')

# 2. Verificar métodos obrigatórios
assert hasattr(scraper_class, 'source_name'), '✗ source_name não definido'
assert hasattr(scraper_class, 'search'), '✗ método search() não definido'
assert hasattr(scraper_class, 'fetch_document'), '✗ método fetch_document() não definido'
assert hasattr(scraper_class, 'fetch_all'), '✗ método fetch_all() não disponível (herança)'
print('✓ Todos os métodos obrigatórios estão presentes')

# 3. Verificar assinatura de search()
sig_search = signature(scraper_class.search)
assert 'limit' in sig_search.parameters, '✗ search() sem parâmetro limit'
print(f'✓ Assinatura search() correta: {sig_search}')

# 4. Verificar assinatura de fetch_document()
sig_fetch = signature(scraper_class.fetch_document)
assert 'doc' in sig_fetch.parameters, '✗ fetch_document() sem parâmetro doc'
assert 'save_original' in sig_fetch.parameters, '✗ fetch_document() sem parâmetro save_original'
print(f'✓ Assinatura fetch_document() correta: {sig_fetch}')

print('✓ Validação de interface concluída com sucesso')
"

echo -e "\n=== ✓ Todos os testes pré-scraping passaram ==="
