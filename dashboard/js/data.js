/**
 * Data loading, parsing, and aggregation utilities.
 * Handles JSON result files and provides computed statistics.
 */

const DataManager = {
  _rawData: null,
  _summaries: [],
  _metadata: null,

  /** Load data from a JSON object */
  loadFromObject(data) {
    this._rawData = data;
    this._metadata = data.metadata || {};
    this._summaries = data.summaries || [];
    return this;
  },

  /** Load data from a File object */
  async loadFromFile(file) {
    const text = await file.text();
    const data = JSON.parse(text);
    return this.loadFromObject(data);
  },

  /** Load sample data from URL */
  async loadSampleData() {
    try {
      const resp = await fetch('../results/sample_results.json');
      if (!resp.ok) throw new Error('Sample data not found');
      const data = await resp.json();
      return this.loadFromObject(data);
    } catch (e) {
      console.warn('Could not load sample data, trying relative path...');
      try {
        const resp = await fetch('./results/sample_results.json');
        const data = await resp.json();
        return this.loadFromObject(data);
      } catch (e2) {
        console.error('Failed to load sample data:', e2);
        return null;
      }
    }
  },

  get metadata() { return this._metadata; },
  get summaries() { return this._summaries; },
  get rawResults() { return this._rawData?.raw_results || []; },
  get isLoaded() { return this._summaries.length > 0; },

  /** Get unique values for a field */
  getUnique(field) {
    const vals = new Set(this._summaries.map(s => s[field]));
    return [...vals].sort();
  },

  get backends() { return this.getUnique('backend'); },
  get quantizations() { return this.getUnique('quantization'); },
  get models() { return this.getUnique('model'); },

  /** Filter summaries by criteria */
  filter({ backend, quantization, model } = {}) {
    let data = [...this._summaries];
    if (backend && backend !== 'all') {
      data = data.filter(s => s.backend === backend);
    }
    if (quantization && quantization !== 'all') {
      data = data.filter(s => s.quantization === quantization);
    }
    if (model && model !== 'all') {
      data = data.filter(s => s.model === model);
    }
    return data;
  },

  /** Get aggregated stats across all summaries */
  getOverviewStats(filteredData) {
    const data = filteredData || this._summaries;
    if (!data.length) return null;

    const tps = data.map(d => d.tokens_per_second.mean);
    const ttft = data.map(d => d.ttft_ms.mean);
    const ram = data.map(d => d.peak_ram_mb.mean);
    const quality = data.map(d => d.quality_score.mean);

    // Find best performers
    const bestTps = data.reduce((a, b) =>
      a.tokens_per_second.mean > b.tokens_per_second.mean ? a : b);
    const bestTtft = data.reduce((a, b) =>
      a.ttft_ms.mean < b.ttft_ms.mean ? a : b);
    const bestRam = data.reduce((a, b) =>
      a.peak_ram_mb.mean < b.peak_ram_mb.mean ? a : b);
    const bestQuality = data.reduce((a, b) =>
      a.quality_score.mean > b.quality_score.mean ? a : b);

    return {
      totalConfigs: data.length,
      totalRuns: data.reduce((sum, d) => sum + d.num_runs, 0),
      avgTps: this._mean(tps),
      maxTps: Math.max(...tps),
      minTtft: Math.min(...ttft),
      avgTtft: this._mean(ttft),
      avgRam: this._mean(ram),
      minRam: Math.min(...ram),
      avgQuality: this._mean(quality),
      maxQuality: Math.max(...quality),
      bestTps,
      bestTtft,
      bestRam,
      bestQuality,
    };
  },

  /** Get data formatted for Chart.js */
  getChartData(field, filteredData) {
    const data = filteredData || this._summaries;
    return {
      labels: data.map(d => this._shortLabel(d)),
      values: data.map(d => {
        const nested = d[field];
        return nested?.mean !== undefined ? nested.mean : nested;
      }),
      backends: data.map(d => d.backend),
    };
  },

  /** Create a short label for charts */
  _shortLabel(d) {
    const model = d.model.replace(/[:\-_]/g, ' ').split(' ').slice(0, 2).join(' ');
    return `${model} (${d.quantization})`;
  },

  /** Mean helper */
  _mean(arr) {
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  },

  /** Get backend color */
  getBackendColor(backend, alpha = 1) {
    const colors = {
      'Ollama': `rgba(0, 212, 255, ${alpha})`,
      'llama.cpp': `rgba(168, 85, 247, ${alpha})`,
      'MLX': `rgba(34, 197, 94, ${alpha})`,
    };
    return colors[backend] || `rgba(200, 200, 200, ${alpha})`;
  },

  /** Get gradient for backend */
  getBackendGradient(ctx, backend) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    const colors = {
      'Ollama': ['rgba(0, 212, 255, 0.8)', 'rgba(0, 212, 255, 0.1)'],
      'llama.cpp': ['rgba(168, 85, 247, 0.8)', 'rgba(168, 85, 247, 0.1)'],
      'MLX': ['rgba(34, 197, 94, 0.8)', 'rgba(34, 197, 94, 0.1)'],
    };
    const [start, end] = colors[backend] || ['rgba(200, 200, 200, 0.8)', 'rgba(200, 200, 200, 0.1)'];
    gradient.addColorStop(0, start);
    gradient.addColorStop(1, end);
    return gradient;
  },

  /** Export data as CSV */
  exportCSV(filteredData) {
    const data = filteredData || this._summaries;
    const headers = ['Backend', 'Model', 'Quantization', 'Size', 'Runs',
      'Tokens/s (mean)', 'Tokens/s (median)', 'TTFT (ms)', 'Peak RAM (MB)',
      'Quality Score', 'Load Time (s)'];
    const rows = data.map(d => [
      d.backend, d.model, d.quantization, d.size_label, d.num_runs,
      d.tokens_per_second.mean.toFixed(1),
      d.tokens_per_second.median.toFixed(1),
      d.ttft_ms.mean.toFixed(0),
      d.peak_ram_mb.mean.toFixed(0),
      d.quality_score.mean.toFixed(1),
      d.model_load_time_s.toFixed(1),
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llm_bench_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  },
};
