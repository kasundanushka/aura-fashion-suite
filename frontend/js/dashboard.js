/**
 * Main Frontend Dashboard Orchestration Controller.
 * Manages state, API communication, product switching, KPI summaries, and uploads.
 */

let state = {
  products: [],
  selectedSku: null,
  selectedCategory: 'ALL',
  forecastWeeks: 8,
  alertsData: null,
  currentProductData: null
};

// ==========================================
// Initialization
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
  initEventListeners();
  await loadInitialDashboardData();
});

function initEventListeners() {
  // Product Selector
  const skuSelect = document.getElementById('skuSelect');
  if (skuSelect) {
    skuSelect.addEventListener('change', (e) => {
      selectProduct(e.target.value);
    });
  }

  // Horizon Selector
  const horizonSelect = document.getElementById('horizonSelect');
  if (horizonSelect) {
    horizonSelect.addEventListener('change', (e) => {
      state.forecastWeeks = parseInt(e.target.value, 10);
      if (state.selectedSku) {
        fetchAndRenderForecast(state.selectedSku, state.forecastWeeks);
      }
    });
  }

  // Category Tabs
  document.querySelectorAll('.category-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      filterProductsByCategory(tab.dataset.category);
    });
  });

  // Upload Modal & In-flight Upload State
  const uploadBtn = document.getElementById('openUploadModalBtn');
  const uploadModal = document.getElementById('uploadModal');
  const closeModalBtn = document.getElementById('closeUploadModalBtn');
  const fileInput = document.getElementById('salesCsvFileInput');

  if (uploadBtn && uploadModal) {
    uploadBtn.addEventListener('click', () => {
      // Ensure clean initial dropzone state when opening
      const dropzone = document.getElementById('uploadDropzone');
      if (dropzone && dropzone.dataset.processing !== 'true' && cachedInitialDropzoneHtml) {
        dropzone.innerHTML = cachedInitialDropzoneHtml;
      }
      uploadModal.classList.add('active');
    });
  }

  if (closeModalBtn && uploadModal) {
    closeModalBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      cancelOrCloseUploadModal();
    });
  }

  // Allow clicking outside modal content or pressing Escape to close/cancel
  if (uploadModal) {
    uploadModal.addEventListener('click', (e) => {
      if (e.target === uploadModal) {
        cancelOrCloseUploadModal();
      }
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && uploadModal && uploadModal.classList.contains('active')) {
      cancelOrCloseUploadModal();
    }
  });

  // File Dropzone with Drag & Drop
  const dropzone = document.getElementById('uploadDropzone');
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', (e) => {
      // Don't trigger browse if currently processing
      if (dropzone.dataset.processing === 'true') return;
      fileInput.click();
    });

    fileInput.addEventListener('change', handleFileUpload);

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropzone.dataset.processing !== 'true') {
          dropzone.style.borderColor = 'var(--accent-primary)';
          dropzone.style.background = 'rgba(99, 102, 241, 0.15)';
        }
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (dropzone.dataset.processing !== 'true') {
          dropzone.style.borderColor = '';
          dropzone.style.background = '';
        }
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (dropzone.dataset.processing === 'true') return;
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileUpload({ target: { files: files } });
      }
    });
  }
}

// ==========================================
// API Calls & Data Fetching
// ==========================================

async function loadInitialDashboardData() {
  showToast('Connecting to Clothing AI Backend...');
  try {
    // 1. Fetch Products
    const prodRes = await fetch('/api/products');
    if (!prodRes.ok) throw new Error('Failed to load products.');
    const prodData = await prodRes.json();
    state.products = prodData.products || [];
    populateSkuDropdown(state.products);

    // 2. Fetch Reorder Alerts & KPIs
    const alertsRes = await fetch('/api/reorder-alerts');
    if (!alertsRes.ok) throw new Error('Failed to load inventory alerts.');
    const alertsData = await alertsRes.json();
    state.alertsData = alertsData;

    renderKpiSummaryCards(alertsData.kpis);
    renderAlertsTable(alertsData.alerts);

    // 3. Render Dynamic Category Tabs based on active dataset
    renderCategoryTabs(state.products);

    // 4. Select Initial Product
    if (state.products.length > 0) {
      selectProduct(state.products[0].sku);
    }

  } catch (err) {
    console.error('Initial load error:', err);
    showToast(`Error: ${err.message}`, true);
  }
}

function renderCategoryTabs(products) {
  const container = document.querySelector('.category-tabs');
  if (!container || !products || products.length === 0) return;

  const rawCats = Array.from(new Set(products.map(p => p.category).filter(Boolean)));
  const categories = ['ALL', ...rawCats];

  container.innerHTML = categories.map(cat =>
    `<button class="category-tab ${cat === state.selectedCategory ? 'active' : ''}" data-category="${cat}">${cat === 'ALL' ? 'All' : cat}</button>`
  ).join('');

  container.querySelectorAll('.category-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      container.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      filterProductsByCategory(tab.dataset.category);
    });
  });
}

function populateSkuDropdown(products) {
  const select = document.getElementById('skuSelect');
  if (!select) return;

  select.innerHTML = products.map(p =>
    `<option value="${p.sku}">[${p.sku}] ${p.name} (${p.category})</option>`
  ).join('');
}

function filterProductsByCategory(category) {
  state.selectedCategory = category;
  const filtered = category === 'ALL'
    ? state.products
    : state.products.filter(p => (p.category || '').toLowerCase() === category.toLowerCase());

  populateSkuDropdown(filtered);
  if (filtered.length > 0) {
    selectProduct(filtered[0].sku);
  }
}

function selectProductFromTable(sku) {
  const skuSelect = document.getElementById('skuSelect');
  if (skuSelect) {
    skuSelect.value = sku;
  }
  selectProduct(sku);
  window.scrollTo({ top: 180, behavior: 'smooth' });
}

async function selectProduct(sku) {
  state.selectedSku = sku;
  const prod = state.products.find(p => p.sku === sku);
  state.currentProductData = prod;

  renderProductDetails(prod);
  await Promise.all([
    fetchAndRenderForecast(sku, state.forecastWeeks),
    fetchAndRenderTrend(sku)
  ]);
}

async function fetchAndRenderForecast(sku, weeks) {
  try {
    const res = await fetch(`/api/forecast/${sku}?weeks=${weeks}`);
    if (!res.ok) throw new Error(`Forecast request failed (${res.status})`);
    const data = await res.json();

    renderDemandForecastChart(data.historical, data.forecast, data.baseline);
    renderModelEvaluationMetrics(data.model_evaluation, data.baseline_evaluation);
  } catch (err) {
    console.error('Forecast fetch error:', err);
    showToast(`Forecast error: ${err.message}`, true);
  }
}

async function fetchAndRenderTrend(sku) {
  try {
    const res = await fetch(`/api/trend/${sku}`);
    if (!res.ok) throw new Error(`Trend request failed (${res.status})`);
    const data = await res.json();

    renderTrendBadge(data);
    updateOptimizationDetails(sku);
  } catch (err) {
    console.error('Trend fetch error:', err);
  }
}

// ==========================================
// UI Render Helpers
// ==========================================

function renderKpiSummaryCards(kpis) {
  if (!kpis) return;
  document.getElementById('kpiUrgentCount').textContent = kpis.urgent_reorders ?? 0;
  document.getElementById('kpiReorderBudget').textContent = `LKR ${(kpis.total_estimated_reorder_budget ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
  document.getElementById('kpiReorderSoon').textContent = kpis.reorder_soon ?? 0;
  document.getElementById('kpiOverstocked').textContent = kpis.overstocked ?? 0;
}

function renderProductDetails(prod) {
  if (!prod) return;
  document.getElementById('productSkuBadge').textContent = prod.sku;
  document.getElementById('productNameTitle').textContent = prod.name;
  document.getElementById('productCategoryText').textContent = prod.category;
  document.getElementById('productSubcategoryText').textContent = prod.subcategory || 'Standard';
  document.getElementById('productPriceText').textContent = `LKR ${parseFloat(prod.retail_price).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById('productMaterialText').textContent = prod.material || 'Standard';
  document.getElementById('productStockText').textContent = `${prod.current_stock} Units`;
}

function renderTrendBadge(trendData) {
  const container = document.getElementById('trendBadgeContainer');
  if (!container) return;

  const status = (trendData.trend_status || 'stable').toLowerCase();
  const icon = status === 'rising' ? '▲' : status === 'declining' ? '▼' : '▬';
  const confPct = Math.round((trendData.confidence_score || 0) * 100);
  const growthRate = (trendData.growth_rate_recent * 100).toFixed(1);

  container.className = `trend-card ${status}`;
  container.innerHTML = `
    <div class="trend-badge-left">
      <div class="trend-icon">${icon}</div>
      <div>
        <div class="trend-title">AI Trend Direction</div>
        <div class="trend-status-text">${status} Trend</div>
      </div>
    </div>
    <div class="trend-score">
      <strong>${confPct}%</strong>
      <span>Confidence (${growthRate}% 8w)</span>
    </div>
  `;
}

function updateOptimizationDetails(sku) {
  if (!state.alertsData || !state.alertsData.alerts) return;
  const alertItem = state.alertsData.alerts.find(a => a.sku === sku);
  if (!alertItem) return;

  document.getElementById('optSafetyStock').textContent = `${alertItem.safety_stock} u`;
  document.getElementById('optReorderPoint').textContent = `${alertItem.reorder_point} u`;
  document.getElementById('optRecommendedQty').textContent = alertItem.recommended_quantity > 0 ? `+${alertItem.recommended_quantity} u` : '0 u';

  const actionElem = document.getElementById('optActionText');
  if (actionElem) {
    actionElem.textContent = alertItem.action_summary;
    actionElem.className = `opt-action-text ${getBadgeClass(alertItem.alert_level)}`;
  }
}

function renderModelEvaluationMetrics(prophetEval, baselineEval) {
  const container = document.getElementById('modelMetricsContainer');
  if (!container || !prophetEval) return;

  const pTest = prophetEval.test_metrics || {};
  const bTest = (baselineEval && baselineEval.test_metrics) || {};

  container.innerHTML = `
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size:12px; margin-top: 10px;">
      <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); padding:10px; border-radius:8px;">
        <strong style="color:var(--accent-primary);">Prophet (Primary)</strong>
        <div>Test RMSE: <strong>${pTest.rmse ?? 'N/A'}</strong></div>
        <div>Test MAPE: <strong>${pTest.mape ?? 'N/A'}%</strong></div>
      </div>
      <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3); padding:10px; border-radius:8px;">
        <strong style="color:var(--status-soon);">Baseline (Linear Reg)</strong>
        <div>Test RMSE: <strong>${bTest.rmse ?? 'N/A'}</strong></div>
        <div>Test MAPE: <strong>${bTest.mape ?? 'N/A'}%</strong></div>
      </div>
    </div>
  `;
}

// ==========================================
// File Upload & Pipeline Ingestion
// ==========================================

let activeUploadController = null;
let activeUploadInterval = null;
let cachedInitialDropzoneHtml = '';

function cancelOrCloseUploadModal() {
  const uploadModal = document.getElementById('uploadModal');
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('salesCsvFileInput');
  const closeBtn = document.getElementById('closeUploadModalBtn');

  // If an upload is currently processing, abort it cleanly
  if (activeUploadController) {
    try {
      activeUploadController.abort();
    } catch (_) {}
    activeUploadController = null;
    showToast('Upload cancelled.');
  }

  if (activeUploadInterval) {
    clearInterval(activeUploadInterval);
    activeUploadInterval = null;
  }

  if (dropzone) {
    dropzone.dataset.processing = 'false';
    dropzone.style.cursor = 'pointer';
    if (cachedInitialDropzoneHtml) {
      dropzone.innerHTML = cachedInitialDropzoneHtml;
    }
  }

  if (closeBtn) {
    closeBtn.style.pointerEvents = 'auto';
    closeBtn.style.cursor = 'pointer';
  }

  if (fileInput) {
    fileInput.value = '';
  }

  if (uploadModal) {
    uploadModal.classList.remove('active');
  }
}

async function handleFileUpload(e) {
  const file = e.target.files && e.target.files[0];
  if (!file) return;

  if (!file.name.toLowerCase().endsWith('.csv')) {
    showToast('Please select a valid .csv file.', true);
    if (e.target) e.target.value = '';
    return;
  }

  const u = JSON.parse(localStorage.getItem('aura_user') || '{}');
  const formData = new FormData();
  formData.append('file', file);
  if (u.email) {
    formData.append('uploaded_by', u.email);
  }

  const uploadModal = document.getElementById('uploadModal');
  const dropzone = document.getElementById('uploadDropzone');
  const closeBtn = document.getElementById('closeUploadModalBtn');
  
  if (!cachedInitialDropzoneHtml && dropzone) {
    cachedInitialDropzoneHtml = dropzone.innerHTML;
  }
  const originalDropzoneHtml = cachedInitialDropzoneHtml || (dropzone ? dropzone.innerHTML : '');

  // Keep close button fully clickable and interactive at all times
  if (closeBtn) {
    closeBtn.style.pointerEvents = 'auto';
    closeBtn.style.cursor = 'pointer';
    closeBtn.title = 'Close / Cancel Upload';
  }

  // Lock dropzone & show rotating loading circle
  if (dropzone) {
    dropzone.dataset.processing = 'true';
    dropzone.style.cursor = 'wait';
    dropzone.innerHTML = `
      <div class="ai-spinner-container">
        <div class="ai-spinner-ring">
          <div class="ai-spinner-circle"></div>
          <div class="ai-spinner-circle-inner"></div>
          <div class="ai-spinner-core"></div>
        </div>
        <div class="ai-spinner-title" id="aiProcessTitle">Analyzing &amp; Ingesting Dataset...</div>
        <div class="ai-spinner-sub" id="aiProcessSub">Normalizing column headers &amp; checking schema...</div>
        <div class="ai-spinner-progress">
          <div class="ai-spinner-progress-bar"></div>
        </div>
      </div>
    `;
  }

  // Dynamic status messages during the analysis delay
  const statusSteps = [
    'Normalizing column headers &amp; checking schema...',
    'Validating transactions &amp; aligning time series...',
    'Generating rolling features &amp; lag indicators...',
    'Retraining Prophet &amp; machine learning forecast models...',
    'Finalizing inventory intelligence &amp; reorder points...'
  ];
  let stepIdx = 0;
  if (activeUploadInterval) clearInterval(activeUploadInterval);
  activeUploadInterval = setInterval(() => {
    stepIdx = (stepIdx + 1) % statusSteps.length;
    const subEl = document.getElementById('aiProcessSub');
    if (subEl) subEl.innerHTML = statusSteps[stepIdx];
  }, 2200);

  showToast(`Uploading '${file.name}' & analyzing data...`);

  activeUploadController = new AbortController();

  try {
    const res = await fetch('/api/upload-sales', {
      method: 'POST',
      body: formData,
      signal: activeUploadController.signal
    });
    const data = await res.json();

    if (activeUploadInterval) {
      clearInterval(activeUploadInterval);
      activeUploadInterval = null;
    }
    activeUploadController = null;

    if (!res.ok) throw new Error(data.error || 'Upload failed');

    // Success state inside dropzone
    if (dropzone) {
      dropzone.innerHTML = `
        <div class="ai-spinner-container">
          <div style="font-size: 52px; margin-bottom: 12px;">✅</div>
          <div class="ai-spinner-title" style="color: #34d399;">Dataset Ingestion Complete!</div>
          <div class="ai-spinner-sub">Processed <strong>${Number(data.row_count || 0).toLocaleString()}</strong> rows. AI forecast models retrained.</div>
        </div>
      `;
    }

    showToast('Success! Pipeline retrained & inventory refreshed.');

    // Wait a brief moment to show the success state, then close modal
    setTimeout(async () => {
      if (uploadModal) uploadModal.classList.remove('active');
      if (dropzone) {
        dropzone.dataset.processing = 'false';
        dropzone.style.cursor = 'pointer';
        dropzone.innerHTML = originalDropzoneHtml;
      }
      if (e.target) e.target.value = '';
      await loadInitialDashboardData();
    }, 1400);

  } catch (err) {
    if (activeUploadInterval) {
      clearInterval(activeUploadInterval);
      activeUploadInterval = null;
    }
    activeUploadController = null;

    // If cancelled by user abort, do not display failure error
    if (err.name === 'AbortError') {
      return;
    }

    showToast(`Upload error: ${err.message}`, true);

    if (dropzone) {
      dropzone.dataset.processing = 'false';
      dropzone.style.cursor = 'pointer';
      dropzone.innerHTML = `
        <div class="ai-spinner-container">
          <div style="font-size: 46px; margin-bottom: 10px;">⚠️</div>
          <div class="ai-spinner-title" style="color: #f87171;">Analysis &amp; Ingestion Error</div>
          <div class="ai-spinner-sub" style="margin-bottom: 16px; color: #e2e8f0;">${err.message}</div>
          <button type="button" class="btn btn-primary" id="retryDropzoneBtn" style="padding: 6px 16px; font-size: 12.5px;">
            🔄 Choose Another File
          </button>
        </div>
      `;

      const retryBtn = document.getElementById('retryDropzoneBtn');
      if (retryBtn) {
        retryBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          dropzone.innerHTML = originalDropzoneHtml;
          const fi = document.getElementById('salesCsvFileInput');
          if (fi) {
            fi.value = '';
            fi.click();
          }
        });
      }
    }

    if (e.target) e.target.value = '';
  }
}

function showToast(message, isError = false) {
  const toast = document.getElementById('toastNotification');
  if (!toast) return;
  toast.textContent = message;
  toast.style.borderColor = isError ? 'var(--status-urgent)' : 'var(--accent-primary)';
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 4000);
}
