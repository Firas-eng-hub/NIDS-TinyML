let chart = null;

async function fetchStats() {
  const res = await fetch('/api/stats');
  return await res.json();
}

async function fetchEvents() {
  const classFilter = document.getElementById('classFilter').value;
  const minConfidence = document.getElementById('minConfidence').value;
  const params = new URLSearchParams();
  params.set('limit', '100');
  if (classFilter) params.set('class', classFilter);
  if (minConfidence) params.set('min_confidence', minConfidence);
  const res = await fetch(`/api/events?${params.toString()}`);
  const data = await res.json();
  return data.events || [];
}

function updateChart(byClass) {
  const labels = Object.keys(byClass);
  const values = Object.values(byClass);

  const ctx = document.getElementById('alertsChart').getContext('2d');

  if (chart) {
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update();
    return;
  }

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Alerts',
          data: values,
          backgroundColor: '#3b82f6'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderTable(events) {
  const tbody = document.getElementById('alertsTable');
  tbody.innerHTML = '';

  events.forEach(e => {
    const tr = document.createElement('tr');
    const topFeatures = e.top_feature_values ? JSON.stringify(e.top_feature_values) : '';

    tr.innerHTML = `
      <td>${e.ts || ''}</td>
      <td>${e.predicted_class || ''}</td>
      <td>${(e.confidence ?? 0).toFixed(3)}</td>
      <td>${e.flow_id || ''}</td>
      <td>${topFeatures}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function refresh() {
  const stats = await fetchStats();
  document.getElementById('totalAlerts').textContent = stats.total || 0;
  updateChart(stats.by_class || {});

  const events = await fetchEvents();
  renderTable(events);
}

document.getElementById('refreshBtn').addEventListener('click', refresh);

document.getElementById('classFilter').addEventListener('change', refresh);
document.getElementById('minConfidence').addEventListener('change', refresh);

setInterval(refresh, 2000);
refresh();
