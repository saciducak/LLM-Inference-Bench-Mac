/**
 * Chart.js visualizations for the benchmark dashboard.
 * Creates and manages all charts with dark theme and animations.
 */

const ChartManager = {
  _charts: {},
  _defaultOptions: null,

  init() {
    // Configure Chart.js defaults for dark theme
    Chart.defaults.color = '#9898b8';
    Chart.defaults.borderColor = 'rgba(100, 100, 255, 0.08)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyle = 'circle';
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(10, 10, 30, 0.95)';
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(100, 100, 255, 0.2)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.titleFont = { weight: '600', size: 13 };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };
    Chart.defaults.animation = {
      duration: 800,
      easing: 'easeOutQuart',
    };
  },

  /** Destroy a chart if it exists */
  _destroy(id) {
    if (this._charts[id]) {
      this._charts[id].destroy();
      delete this._charts[id];
    }
  },

  /** Create all charts from filtered data */
  renderAll(filteredData) {
    this.renderThroughputChart(filteredData);
    this.renderTTFTChart(filteredData);
    this.renderRAMChart(filteredData);
    this.renderQualityVsSpeedChart(filteredData);
    this.renderRadarChart(filteredData);
    this.renderLoadTimeChart(filteredData);
  },

  /** Chart 1: Tokens/sec bar chart */
  renderThroughputChart(data) {
    this._destroy('throughput');
    const ctx = document.getElementById('chartThroughput');
    if (!ctx) return;

    const sorted = [...data].sort((a, b) => b.tokens_per_second.mean - a.tokens_per_second.mean);
    const labels = sorted.map(d => DataManager._shortLabel(d));
    const values = sorted.map(d => d.tokens_per_second.mean);
    const errors = sorted.map(d => d.tokens_per_second.std || 0);
    const bgColors = sorted.map(d => DataManager.getBackendColor(d.backend, 0.75));
    const borderColors = sorted.map(d => DataManager.getBackendColor(d.backend, 1));

    this._charts['throughput'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Tokens/sec',
          data: values,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const i = ctx.dataIndex;
                return `± ${errors[i].toFixed(1)} std | Backend: ${sorted[i].backend}`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Tokens / Second', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11 } },
          }
        }
      }
    });
  },

  /** Chart 2: TTFT comparison */
  renderTTFTChart(data) {
    this._destroy('ttft');
    const ctx = document.getElementById('chartTTFT');
    if (!ctx) return;

    const sorted = [...data].sort((a, b) => a.ttft_ms.mean - b.ttft_ms.mean);
    const labels = sorted.map(d => DataManager._shortLabel(d));
    const values = sorted.map(d => d.ttft_ms.mean);
    const bgColors = sorted.map(d => DataManager.getBackendColor(d.backend, 0.75));
    const borderColors = sorted.map(d => DataManager.getBackendColor(d.backend, 1));

    this._charts['ttft'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Time to First Token (ms)',
          data: values,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => `Backend: ${sorted[ctx.dataIndex].backend}`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'TTFT (ms)', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
          },
          x: {
            grid: { display: false },
            ticks: { maxRotation: 45, font: { size: 10 } },
          }
        }
      }
    });
  },

  /** Chart 3: RAM usage horizontal bar */
  renderRAMChart(data) {
    this._destroy('ram');
    const ctx = document.getElementById('chartRAM');
    if (!ctx) return;

    const sorted = [...data].sort((a, b) => a.peak_ram_mb.mean - b.peak_ram_mb.mean);
    const labels = sorted.map(d => DataManager._shortLabel(d));
    const values = sorted.map(d => d.peak_ram_mb.mean);
    const bgColors = sorted.map(d => DataManager.getBackendColor(d.backend, 0.6));
    const borderColors = sorted.map(d => DataManager.getBackendColor(d.backend, 1));

    this._charts['ram'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Peak RAM (MB)',
          data: values,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw.toFixed(0)} MB`,
              afterLabel: (ctx) => {
                const totalRam = DataManager.metadata?.system?.total_ram_gb || 16;
                const pct = ((ctx.raw / (totalRam * 1024)) * 100).toFixed(1);
                return `${pct}% of ${totalRam}GB total`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Peak RAM (MB)', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11 } },
          }
        }
      }
    });
  },

  /** Chart 4: Quality vs Speed scatter */
  renderQualityVsSpeedChart(data) {
    this._destroy('qvs');
    const ctx = document.getElementById('chartQualityVsSpeed');
    if (!ctx) return;

    // Group by backend
    const backends = {};
    data.forEach(d => {
      if (!backends[d.backend]) backends[d.backend] = [];
      backends[d.backend].push(d);
    });

    const datasets = Object.entries(backends).map(([name, items]) => ({
      label: name,
      data: items.map(d => ({
        x: d.tokens_per_second.mean,
        y: d.quality_score.mean,
        r: Math.max(6, Math.min(20, d.peak_ram_mb.mean / 150)),
        model: d.model,
        quantization: d.quantization,
      })),
      backgroundColor: DataManager.getBackendColor(name, 0.6),
      borderColor: DataManager.getBackendColor(name, 1),
      borderWidth: 2,
    }));

    this._charts['qvs'] = new Chart(ctx, {
      type: 'bubble',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const d = ctx.raw;
                return [
                  `${d.model} (${d.quantization})`,
                  `Speed: ${d.x.toFixed(1)} tok/s`,
                  `Quality: ${d.y.toFixed(0)}`,
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Tokens/sec →', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
          },
          y: {
            title: { display: true, text: 'Quality Score →', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
            min: 0, max: 100,
          }
        }
      }
    });
  },

  /** Chart 5: Radar chart for multi-metric comparison */
  renderRadarChart(data) {
    this._destroy('radar');
    const ctx = document.getElementById('chartRadar');
    if (!ctx) return;

    // Take top 5 configs
    const top = [...data]
      .sort((a, b) => b.tokens_per_second.mean - a.tokens_per_second.mean)
      .slice(0, 5);

    // Normalize each metric to 0-100
    const maxTps = Math.max(...data.map(d => d.tokens_per_second.mean));
    const maxTtft = Math.max(...data.map(d => d.ttft_ms.mean));
    const maxRam = Math.max(...data.map(d => d.peak_ram_mb.mean));

    const datasets = top.map(d => ({
      label: DataManager._shortLabel(d),
      data: [
        (d.tokens_per_second.mean / maxTps) * 100,         // Speed
        ((maxTtft - d.ttft_ms.mean) / maxTtft) * 100,      // Responsiveness (inverted)
        ((maxRam - d.peak_ram_mb.mean) / maxRam) * 100,     // Efficiency (inverted)
        d.quality_score.mean,                                // Quality
        Math.min(100, (d.tokens_per_second.mean / (d.peak_ram_mb.mean / 1000)) * 10), // Perf/RAM ratio
      ],
      backgroundColor: DataManager.getBackendColor(d.backend, 0.15),
      borderColor: DataManager.getBackendColor(d.backend, 0.8),
      borderWidth: 2,
      pointBackgroundColor: DataManager.getBackendColor(d.backend, 1),
      pointRadius: 4,
    }));

    this._charts['radar'] = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Speed', 'Responsiveness', 'RAM Efficiency', 'Quality', 'Perf/RAM'],
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: {
              stepSize: 25,
              color: '#6868a0',
              backdropColor: 'transparent',
              font: { size: 10 },
            },
            grid: { color: 'rgba(100, 100, 255, 0.08)' },
            pointLabels: {
              color: '#9898b8',
              font: { size: 11, weight: '500' },
            },
            angleLines: { color: 'rgba(100, 100, 255, 0.08)' },
          }
        },
        plugins: {
          legend: {
            position: 'bottom',
            labels: { font: { size: 11 } },
          },
        }
      }
    });
  },

  /** Chart 6: Model load time comparison */
  renderLoadTimeChart(data) {
    this._destroy('loadtime');
    const ctx = document.getElementById('chartLoadTime');
    if (!ctx) return;

    const sorted = [...data].sort((a, b) => a.model_load_time_s - b.model_load_time_s);
    const labels = sorted.map(d => DataManager._shortLabel(d));
    const values = sorted.map(d => d.model_load_time_s);
    const bgColors = sorted.map(d => DataManager.getBackendColor(d.backend, 0.6));
    const borderColors = sorted.map(d => DataManager.getBackendColor(d.backend, 1));

    this._charts['loadtime'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Model Load Time (s)',
          data: values,
          backgroundColor: bgColors,
          borderColor: borderColors,
          borderWidth: 1.5,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw.toFixed(1)}s`,
              afterLabel: (ctx) => `Backend: ${sorted[ctx.dataIndex].backend}`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: 'Load Time (seconds)', color: '#6868a0' },
            grid: { color: 'rgba(100, 100, 255, 0.06)' },
          },
          x: {
            grid: { display: false },
            ticks: { maxRotation: 45, font: { size: 10 } },
          }
        }
      }
    });
  },
};
