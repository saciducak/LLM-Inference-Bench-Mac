/**
 * Main application logic for the LLM Inference Bench dashboard.
 * Orchestrates data loading, filtering, rendering, and user interactions.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ─── State ─────────────────────────────────────────────
  let currentFilters = {
    backend: 'all',
    quantization: 'all',
  };

  // ─── DOM References ────────────────────────────────────
  const dom = {
    fileInput: document.getElementById('fileInput'),
    loadSampleBtn: document.getElementById('loadSampleBtn'),
    backendFilter: document.getElementById('filterBackend'),
    quantFilter: document.getElementById('filterQuant'),
    exportCsvBtn: document.getElementById('exportCsv'),
    exportPngBtn: document.getElementById('exportPng'),

    uploadZone: document.getElementById('uploadZone'),
    dashboardContent: document.getElementById('dashboardContent'),
    systemInfo: document.getElementById('systemInfo'),

    // Metric cards
    metricTps: document.getElementById('metricTps'),
    metricTtft: document.getElementById('metricTtft'),
    metricRam: document.getElementById('metricRam'),
    metricQuality: document.getElementById('metricQuality'),
    metricConfigs: document.getElementById('metricConfigs'),
    metricRuns: document.getElementById('metricRuns'),

    // Table
    tableBody: document.getElementById('resultsTableBody'),
  };

  // ─── Initialize ────────────────────────────────────────
  ChartManager.init();
  setupEventListeners();
  
  // Auto-load sample data
  loadSampleData();

  // ─── Event Listeners ──────────────────────────────────
  function setupEventListeners() {
    // File input
    dom.fileInput?.addEventListener('change', handleFileUpload);

    // Load sample data
    dom.loadSampleBtn?.addEventListener('click', loadSampleData);

    // Filters
    dom.backendFilter?.addEventListener('change', handleFilterChange);
    dom.quantFilter?.addEventListener('change', handleFilterChange);

    // Export
    dom.exportCsvBtn?.addEventListener('click', () => {
      const filtered = DataManager.filter(currentFilters);
      DataManager.exportCSV(filtered);
      showToast('CSV exported successfully!');
    });

    dom.exportPngBtn?.addEventListener('click', exportChartsPNG);

    // Upload zone drag & drop
    if (dom.uploadZone) {
      dom.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.add('upload-zone--active');
      });
      dom.uploadZone.addEventListener('dragleave', () => {
        dom.uploadZone.classList.remove('upload-zone--active');
      });
      dom.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.uploadZone.classList.remove('upload-zone--active');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.json')) {
          loadFile(file);
        }
      });
      dom.uploadZone.addEventListener('click', () => dom.fileInput?.click());
    }

    // Table sorting
    document.querySelectorAll('.data-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => handleSort(th));
    });
  }

  // ─── Data Loading ─────────────────────────────────────
  async function loadSampleData() {
    try {
      await DataManager.loadSampleData();
      if (DataManager.isLoaded) {
        onDataLoaded();
        showToast('Sample data loaded');
      }
    } catch (e) {
      console.error('Failed to load sample data:', e);
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (file) await loadFile(file);
  }

  async function loadFile(file) {
    try {
      await DataManager.loadFromFile(file);
      onDataLoaded();
      showToast(`Loaded: ${file.name}`);
    } catch (e) {
      showToast(`Error: ${e.message}`, true);
    }
  }

  // ─── Data Loaded Handler ──────────────────────────────
  function onDataLoaded() {
    // Show dashboard, hide upload
    if (dom.uploadZone) dom.uploadZone.classList.add('hidden');
    if (dom.dashboardContent) dom.dashboardContent.classList.remove('hidden');

    // Populate filters
    populateFilters();

    // Update system info
    updateSystemInfo();

    // Render everything
    renderDashboard();
  }

  function populateFilters() {
    if (dom.backendFilter) {
      const backends = DataManager.backends;
      dom.backendFilter.innerHTML = '<option value="all">All Backends</option>';
      backends.forEach(b => {
        dom.backendFilter.innerHTML += `<option value="${b}">${b}</option>`;
      });
    }

    if (dom.quantFilter) {
      const quants = DataManager.quantizations;
      dom.quantFilter.innerHTML = '<option value="all">All Quantizations</option>';
      quants.forEach(q => {
        dom.quantFilter.innerHTML += `<option value="${q}">${q}</option>`;
      });
    }
  }

  function updateSystemInfo() {
    const sys = DataManager.metadata?.system;
    if (sys && dom.systemInfo) {
      dom.systemInfo.textContent = `${sys.chip} · ${sys.total_ram_gb}GB RAM · ${sys.os_version}`;
    }
  }

  // ─── Render Dashboard ─────────────────────────────────
  function renderDashboard() {
    const filtered = DataManager.filter(currentFilters);
    const stats = DataManager.getOverviewStats(filtered);

    if (!stats) return;

    // Update metric cards
    updateMetricCards(stats);

    // Render charts
    ChartManager.renderAll(filtered);

    // Render table
    renderTable(filtered, stats);
  }

  function updateMetricCards(stats) {
    if (dom.metricTps) {
      dom.metricTps.querySelector('.metric-card__value').innerHTML =
        `${stats.maxTps.toFixed(1)} <span>tok/s</span>`;
      dom.metricTps.querySelector('.metric-card__detail').textContent =
        `Best: ${stats.bestTps.model} (${stats.bestTps.backend})`;
    }
    if (dom.metricTtft) {
      dom.metricTtft.querySelector('.metric-card__value').innerHTML =
        `${stats.minTtft.toFixed(0)} <span>ms</span>`;
      dom.metricTtft.querySelector('.metric-card__detail').textContent =
        `Best: ${stats.bestTtft.model} (${stats.bestTtft.backend})`;
    }
    if (dom.metricRam) {
      dom.metricRam.querySelector('.metric-card__value').innerHTML =
        `${stats.minRam.toFixed(0)} <span>MB</span>`;
      dom.metricRam.querySelector('.metric-card__detail').textContent =
        `Most efficient: ${stats.bestRam.model}`;
    }
    if (dom.metricQuality) {
      dom.metricQuality.querySelector('.metric-card__value').innerHTML =
        `${stats.maxQuality.toFixed(0)} <span>/100</span>`;
      dom.metricQuality.querySelector('.metric-card__detail').textContent =
        `Best: ${stats.bestQuality.model}`;
    }
    if (dom.metricConfigs) {
      dom.metricConfigs.querySelector('.metric-card__value').textContent =
        stats.totalConfigs;
    }
    if (dom.metricRuns) {
      dom.metricRuns.querySelector('.metric-card__value').textContent =
        stats.totalRuns;
    }
  }

  // ─── Table Rendering ──────────────────────────────────
  function renderTable(data, stats) {
    if (!dom.tableBody) return;

    // Find best values for highlighting
    const bestTps = Math.max(...data.map(d => d.tokens_per_second.mean));
    const bestTtft = Math.min(...data.map(d => d.ttft_ms.mean));
    const bestRam = Math.min(...data.map(d => d.peak_ram_mb.mean));
    const bestQual = Math.max(...data.map(d => d.quality_score.mean));

    const sorted = [...data].sort((a, b) =>
      b.tokens_per_second.mean - a.tokens_per_second.mean);

    dom.tableBody.innerHTML = sorted.map((d, i) => {
      const badgeClass = d.backend === 'Ollama' ? 'badge--ollama' :
                         d.backend === 'llama.cpp' ? 'badge--llamacpp' :
                         'badge--mlx';

      const qualClass = d.quality_score.mean >= 75 ? 'score-bar__fill--high' :
                        d.quality_score.mean >= 50 ? 'score-bar__fill--medium' :
                        'score-bar__fill--low';

      const tpsClass = d.tokens_per_second.mean === bestTps ? 'best-value' : '';
      const ttftClass = d.ttft_ms.mean === bestTtft ? 'best-value' : '';
      const ramClass = d.peak_ram_mb.mean === bestRam ? 'best-value' : '';

      return `<tr>
        <td>${i + 1}</td>
        <td><span class="badge ${badgeClass}">${d.backend}</span></td>
        <td>${d.model}</td>
        <td><code>${d.quantization}</code></td>
        <td>${d.size_label}</td>
        <td class="${tpsClass}">${d.tokens_per_second.mean.toFixed(1)}</td>
        <td class="${ttftClass}">${d.ttft_ms.mean.toFixed(0)}</td>
        <td class="${ramClass}">${d.peak_ram_mb.mean.toFixed(0)}</td>
        <td>
          <div class="score-bar">
            <span>${d.quality_score.mean.toFixed(0)}</span>
            <div class="score-bar__track">
              <div class="score-bar__fill ${qualClass}" style="width: ${d.quality_score.mean}%"></div>
            </div>
          </div>
        </td>
        <td>${d.model_load_time_s.toFixed(1)}s</td>
      </tr>`;
    }).join('');
  }

  // ─── Table Sorting ────────────────────────────────────
  let sortState = { field: null, direction: 'desc' };

  function handleSort(th) {
    const field = th.dataset.sort;

    // Toggle direction
    if (sortState.field === field) {
      sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
    } else {
      sortState.field = field;
      sortState.direction = 'desc';
    }

    // Update header styles
    document.querySelectorAll('.data-table th').forEach(h => {
      h.classList.remove('sorted-asc', 'sorted-desc');
    });
    th.classList.add(`sorted-${sortState.direction}`);

    // Re-render
    renderDashboard();
  }

  // ─── Filter Handling ──────────────────────────────────
  function handleFilterChange() {
    currentFilters = {
      backend: dom.backendFilter?.value || 'all',
      quantization: dom.quantFilter?.value || 'all',
    };
    renderDashboard();
  }

  // ─── Export PNG ───────────────────────────────────────
  function exportChartsPNG() {
    const chartIds = ['chartThroughput', 'chartTTFT', 'chartRAM',
                      'chartQualityVsSpeed', 'chartRadar', 'chartLoadTime'];
    chartIds.forEach(id => {
      const canvas = document.getElementById(id);
      if (canvas) {
        const link = document.createElement('a');
        link.download = `${id}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      }
    });
    showToast('Charts exported as PNG');
  }

  // ─── Toast Notification ───────────────────────────────
  function showToast(message, isError = false) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    if (isError) {
      toast.style.borderColor = 'var(--accent-red)';
    }
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
  }
});
