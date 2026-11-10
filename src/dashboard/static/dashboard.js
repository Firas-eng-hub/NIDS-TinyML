/* NIDS Dashboard – enhanced UI */

const CLASS_COLORS = {
  Normal:     '#3fb950',
  DoS:        '#f85149',
  BruteForce: '#d29922',
  Botnet:     '#bc8cff',
  WebAttack:  '#ff7b72',
};

function classColor(cls) {
  return CLASS_COLORS[cls] || '#8b949e';
}

function badgeClass(cls) {
  const map = {
    Normal: 'badge-Normal',
    DoS: 'badge-DoS',
    BruteForce: 'badge-BruteForce',
    Botnet: 'badge-Botnet',
    WebAttack: 'badge-WebAttack',
  };
  return map[cls] || 'badge-Unknown';
}

// ── Chart ────────────────────────────────────────────────────────────────────
let chart = null;

function updateChart(byClass) {
  const labels = Object.keys(byClass);
  const values = Object.values(byClass);
  const colors = labels.map(classColor);

  // Legend
  const legendEl = document.getElementById('chartLegend');
  legendEl.innerHTML = labels.map(l => `
    <div class="legend-item">
      <div class="legend-dot" style="background:${classColor(l)}"></div>
      ${l}
    </div>
  `).join('');

  const ctx = document.getElementById('alertsChart').getContext('2d');

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[0].backgroundColor = colors.map(c => c + '33');
    chart.data.datasets[0].borderColor = colors;
    chart.update();
    return;
  }

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Alerts',
        data: values,
        backgroundColor: colors.map(c => c + '33'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1c2333',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#e6edf3',
          bodyColor: '#8b949e',
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(48,54,61,.5)' },
          ticks: { color: '#8b949e' },
        },
        y: {
          grid: { color: 'rgba(48,54,61,.5)' },
          ticks: { color: '#8b949e', precision: 0 },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Class breakdown panel ────────────────────────────────────────────────────
function updateBreakdown(byClass, total) {
  const el = document.getElementById('classBreakdown');
  if (!Object.keys(byClass).length) {
    el.innerHTML = '<div style="padding:24px;color:var(--muted);font-size:13px;">No data yet</div>';
    return;
  }

  const sorted = Object.entries(byClass).sort((a, b) => b[1] - a[1]);
  el.innerHTML = sorted.map(([cls, count]) => {
    const pct = total ? ((count / total) * 100).toFixed(1) : 0;
    const color = classColor(cls);
    return `
      <div style="padding:13px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;">
        <div style="width:10px;height:10px;border-radius:50%;background:${color};flex-shrink:0;"></div>
        <span style="flex:1;font-size:13px;">${cls}</span>
        <span style="font-size:12px;color:var(--muted);width:36px;text-align:right;">${pct}%</span>
        <span style="font-size:14px;font-weight:600;width:42px;text-align:right;">${count}</span>
      </div>
      <div style="height:3px;background:var(--surface2);margin:0 18px;">
        <div style="width:${pct}%;height:100%;background:${color};border-radius:2px;transition:width .4s;"></div>
      </div>
    `;
  }).join('');
}

// ── Stat cards ───────────────────────────────────────────────────────────────
function updateStats(byClass, total) {
  const attacks = total - (byClass['Normal'] || 0);
  const normal  = byClass['Normal'] || 0;
  const rate    = total ? ((attacks / total) * 100).toFixed(1) + '%' : '0%';

  document.getElementById('statTotal').textContent   = total;
  document.getElementById('statAttacks').textContent = attacks;
  document.getElementById('statNormal').textContent  = normal;
  document.getElementById('statRate').textContent    = rate;
}

// ── Table ────────────────────────────────────────────────────────────────────
function confidenceBar(conf) {
  const pct  = Math.round(conf * 100);
  const color = conf > 0.8 ? '#f85149' : conf > 0.5 ? '#d29922' : '#3fb950';
  return `
    <div class="conf-wrap">
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:${pct}%;background:${color};"></div>
      </div>
      <span class="conf-val">${conf.toFixed(3)}</span>
    </div>
  `;
}

function formatFeatures(topFeatureValues) {
  if (!topFeatureValues) return '<span class="features-cell">—</span>';
  const entries = Object.entries(topFeatureValues)
    .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(2) : v}`)
    .join('  ·  ');
  return `<div class="features-cell" title="${entries}">${entries}</div>`;
}

function renderTable(events) {
  const tbody = document.getElementById('alertsTable');
  document.getElementById('tableCount').textContent = `${events.length} events`;

  if (!events.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="5">No alerts match the current filters</td></tr>';
    return;
  }

  tbody.innerHTML = events.map(e => {
    const cls  = e.predicted_class || 'Unknown';
    const conf = e.confidence ?? 0;
    return `
      <tr>
        <td class="ts">${e.ts || '—'}</td>
        <td><span class="badge ${badgeClass(cls)}">${cls}</span></td>
        <td>${confidenceBar(conf)}</td>
        <td style="font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;">${e.flow_id || '—'}</td>
        <td>${formatFeatures(e.top_feature_values)}</td>
      </tr>
    `;
  }).join('');
}

// ── API ──────────────────────────────────────────────────────────────────────
async function fetchStats() {
  const res = await fetch('/api/stats');
  return res.json();
}

async function fetchEvents() {
  const cls    = document.getElementById('classFilter').value;
  const minConf = document.getElementById('minConfidence').value;
  const params = new URLSearchParams({ limit: '100' });
  if (cls)      params.set('class', cls);
  if (minConf)  params.set('min_confidence', minConf);
  const res = await fetch(`/api/events?${params}`);
  const data = await res.json();
  return data.events || [];
}

// ── Refresh loop ─────────────────────────────────────────────────────────────
const REFRESH_INTERVAL = 2000;   // ms
let nextRefresh = Date.now();

function updateCountdown() {
  const remaining = Math.max(0, Math.ceil((nextRefresh - Date.now()) / 1000));
  document.getElementById('countdown').textContent =
    remaining > 0 ? `next update in ${remaining}s` : '';
}

async function refresh() {
  try {
    const [stats, events] = await Promise.all([fetchStats(), fetchEvents()]);
    const byClass = stats.by_class || {};
    const total   = stats.total   || 0;

    updateStats(byClass, total);
    updateChart(byClass);
    updateBreakdown(byClass, total);
    renderTable(events);
  } catch (err) {
    console.error('Dashboard refresh failed:', err);
  }

  nextRefresh = Date.now() + REFRESH_INTERVAL;
}

document.getElementById('refreshBtn').addEventListener('click', refresh);
document.getElementById('classFilter').addEventListener('change', refresh);
document.getElementById('minConfidence').addEventListener('change', refresh);

setInterval(refresh, REFRESH_INTERVAL);
setInterval(updateCountdown, 500);
refresh();
