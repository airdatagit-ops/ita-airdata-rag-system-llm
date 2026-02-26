// Aviation RAG Web App - Custom JavaScript
// Este arquivo é opcional e pode ser usado para adicionar funcionalidades extras

/**
 * Form validation for search page
 */
function validateSearchForm() {
  const form = document.getElementById('search-form');
  
  if (form) {
    form.addEventListener('submit', function(e) {
      const query = document.getElementById('query').value.trim();
      
      if (query.length < 3) {
        e.preventDefault();
        alert('Por favor, digite uma pergunta com pelo menos 3 caracteres.');
        return false;
      }
      
      if (query.length > 1000) {
        e.preventDefault();
        alert('Pergunta muito longa. Máximo de 1000 caracteres.');
        return false;
      }
      
      // Show loading state
      const submitBtn = form.querySelector('button[type="submit"]');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '⏳ Buscando...';
      }
    });
  }
}

/**
 * Auto-refresh stats page
 */
function autoRefreshStats(intervalMinutes = 5) {
  if (window.location.pathname === '/stats') {
    const interval = intervalMinutes * 60 * 1000; // Convert to milliseconds
    
    setInterval(() => {
      console.log('Auto-refreshing stats...');
      window.location.reload();
    }, interval);
  }
}

/**
 * Copy source text to clipboard
 */
function addCopyButtons() {
  const sourceTexts = document.querySelectorAll('.source-text');
  
  sourceTexts.forEach((sourceText, index) => {
    const copyBtn = document.createElement('button');
    copyBtn.innerHTML = '📋 Copiar';
    copyBtn.className = 'copy-btn';
    copyBtn.style.cssText = 'margin-top: 0.5rem; padding: 4px 8px; background: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 0.875rem;';
    
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(sourceText.textContent)
        .then(() => {
          copyBtn.innerHTML = '✅ Copiado!';
          setTimeout(() => {
            copyBtn.innerHTML = '📋 Copiar';
          }, 2000);
        })
        .catch(err => {
          console.error('Erro ao copiar:', err);
          alert('Erro ao copiar texto');
        });
    });
    
    sourceText.parentElement.insertBefore(copyBtn, sourceText.nextSibling);
  });
}

/**
 * Highlight search terms in results
 */
function highlightSearchTerms() {
  const urlParams = new URLSearchParams(window.location.search);
  const query = urlParams.get('query');
  
  if (query && query.length > 3) {
    const terms = query.toLowerCase().split(' ').filter(t => t.length > 3);
    const sourceTexts = document.querySelectorAll('.source-text');
    
    sourceTexts.forEach(sourceText => {
      let html = sourceText.innerHTML;
      
      terms.forEach(term => {
        const regex = new RegExp(`(${term})`, 'gi');
        html = html.replace(regex, '<mark style="background-color: #fff59d;">$1</mark>');
      });
      
      sourceText.innerHTML = html;
    });
  }
}

/**
 * Smooth scroll to results
 */
function smoothScrollToResults() {
  const resultsSection = document.querySelector('.results-section');
  
  if (resultsSection) {
    setTimeout(() => {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }
}

/**
 * Format numbers with locale
 */
function formatNumbers() {
  const numberElements = document.querySelectorAll('[data-number]');
  
  numberElements.forEach(el => {
    const number = parseInt(el.textContent.replace(/[^\d]/g, ''));
    el.textContent = number.toLocaleString('pt-BR');
  });
}

/**
 * Show/hide advanced search options
 */
function toggleAdvancedOptions() {
  const toggleBtn = document.getElementById('toggle-advanced');
  const advancedOptions = document.getElementById('advanced-options');
  
  if (toggleBtn && advancedOptions) {
    toggleBtn.addEventListener('click', () => {
      const isHidden = advancedOptions.style.display === 'none';
      advancedOptions.style.display = isHidden ? 'block' : 'none';
      toggleBtn.textContent = isHidden ? '▼ Ocultar opções avançadas' : '▶ Mostrar opções avançadas';
    });
  }
}

/**
 * Add keyboard shortcuts
 */
function addKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K: Focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      const searchInput = document.getElementById('query');
      if (searchInput) {
        searchInput.focus();
      }
    }
    
    // Ctrl/Cmd + R: Refresh stats
    if ((e.ctrlKey || e.metaKey) && e.key === 'r' && window.location.pathname === '/stats') {
      e.preventDefault();
      window.location.reload();
    }
  });
}

/**
 * Initialize tooltips
 */
function initTooltips() {
  const tooltipElements = document.querySelectorAll('[data-tooltip]');
  
  tooltipElements.forEach(el => {
    el.addEventListener('mouseenter', (e) => {
      const tooltip = document.createElement('div');
      tooltip.className = 'tooltip';
      tooltip.textContent = el.getAttribute('data-tooltip');
      tooltip.style.cssText = 'position: absolute; background: #333; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.875rem; z-index: 1000; white-space: nowrap;';
      
      document.body.appendChild(tooltip);
      
      const rect = el.getBoundingClientRect();
      tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
      tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
      
      el._tooltip = tooltip;
    });
    
    el.addEventListener('mouseleave', (e) => {
      if (el._tooltip) {
        el._tooltip.remove();
        el._tooltip = null;
      }
    });
  });
}

/**
 * Track search analytics (example)
 */
function trackSearch(query, resultsCount) {
  // Example: send to analytics service
  console.log('Search tracked:', { query, resultsCount, timestamp: new Date() });
  
  // If you use Google Analytics:
  // gtag('event', 'search', { search_term: query, results_count: resultsCount });
}

/**
 * Initialize all features
 */
document.addEventListener('DOMContentLoaded', () => {
  console.log('🛩️ Aviation RAG Web App initialized');
  
  // Initialize features
  validateSearchForm();
  addCopyButtons();
  smoothScrollToResults();
  formatNumbers();
  toggleAdvancedOptions();
  addKeyboardShortcuts();
  initTooltips();
  
  // Optional: auto-refresh stats every 5 minutes
  // autoRefreshStats(5);
});

/**
 * Export functions for external use
 */
window.AviationRAG = {
  validateSearchForm,
  autoRefreshStats,
  addCopyButtons,
  highlightSearchTerms,
  smoothScrollToResults,
  formatNumbers,
  trackSearch
};
