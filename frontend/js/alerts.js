/**
 * Reorder Alerts Table & Filtering Logic.
 * Handles table rendering, column sorting, urgency filtering, and row interactions.
 */

let allAlertsData = [];
let currentFilter = 'ALL';
let currentSortColumn = 'urgency';
let isSortAsc = true;

function renderAlertsTable(alerts) {
  allAlertsData = alerts || [];
  const tbody = document.getElementById('alertsTableBody');
  if (!tbody) return;

  // Filter alerts
  let filtered = allAlertsData;
  if (currentFilter !== 'ALL') {
    filtered = allAlertsData.filter(a => a.alert_level === currentFilter);
  }

  // Sort alerts
  filtered.sort((a, b) => {
    let valA, valB;
    if (currentSortColumn === 'sku') {
      valA = a.sku; valB = b.sku;
      return isSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    } else if (currentSortColumn === 'name') {
      valA = a.name || ''; valB = b.name || '';
      return isSortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
    } else if (currentSortColumn === 'stock') {
      valA = a.current_stock; valB = b.current_stock;
      return isSortAsc ? valA - valB : valB - valA;
    } else if (currentSortColumn === 'forecast') {
      valA = a.forecasted_demand_total; valB = b.forecasted_demand_total;
      return isSortAsc ? valA - valB : valB - valA;
    } else if (currentSortColumn === 'reorder') {
      valA = a.recommended_quantity; valB = b.recommended_quantity;
      return isSortAsc ? valA - valB : valB - valA;
    } else if (currentSortColumn === 'cost') {
      valA = a.estimated_reorder_cost || 0; valB = b.estimated_reorder_cost || 0;
      return isSortAsc ? valA - valB : valB - valA;
    } else {
      // Default: priority map
      const priority = { 'URGENT_REORDER': 0, 'REORDER_SOON': 1, 'OVERSTOCKED': 2, 'SUFFICIENT_STOCK': 3 };
      valA = priority[a.alert_level] ?? 9;
      valB = priority[b.alert_level] ?? 9;
      return isSortAsc ? valA - valB : valB - valA;
    }
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 24px; color: var(--text-muted);">No items match the selected filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(item => {
    const badgeClass = getBadgeClass(item.alert_level);
    const trendClass = `trend-card ${item.trend_status.toLowerCase()}`;
    const trendIcon = item.trend_status === 'rising' ? '▲' : item.trend_status === 'declining' ? '▼' : '▬';

    return `
      <tr onclick="selectProductFromTable('${item.sku}')" style="cursor:pointer;" title="Click to view forecast for ${item.sku}">
        <td><strong style="color:var(--accent-primary);">${item.sku}</strong></td>
        <td>
          <div style="font-weight:600; color:var(--text-primary);">${item.name || item.sku}</div>
          <div style="font-size:11px; color:var(--text-muted);">${item.category} • ${item.subcategory || ''}</div>
        </td>
        <td>
          <span class="status-badge ${badgeClass}">
            ${item.alert_level.replace('_', ' ')}
          </span>
        </td>
        <td><strong>${item.current_stock}</strong> <span style="font-size:11px; color:var(--text-muted);">(SS: ${item.safety_stock})</span></td>
        <td>${item.forecasted_demand_total} units</td>
        <td>
          <span style="font-weight:600; font-size:12px; color: ${item.trend_status === 'rising' ? '#10b981' : item.trend_status === 'declining' ? '#f43f5e' : '#8b5cf6'}">
            ${trendIcon} ${item.trend_status} (${Math.round(item.trend_confidence * 100)}%)
          </span>
        </td>
        <td>
          <strong style="color:${item.recommended_quantity > 0 ? 'var(--status-urgent)' : 'var(--text-secondary)'}">
            ${item.recommended_quantity > 0 ? `+${item.recommended_quantity}` : '0'}
          </strong>
        </td>
        <td>LKR ${(item.estimated_reorder_cost || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
      </tr>
    `;
  }).join('');
}

function getBadgeClass(level) {
  switch (level) {
    case 'URGENT_REORDER': return 'badge-urgent';
    case 'REORDER_SOON': return 'badge-soon';
    case 'OVERSTOCKED': return 'badge-overstocked';
    case 'SUFFICIENT_STOCK': return 'badge-sufficient';
    default: return 'badge-soon';
  }
}

function setAlertFilter(filterType) {
  currentFilter = filterType;
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.filter === filterType);
  });
  renderAlertsTable(allAlertsData);
}

function sortAlertsTable(colKey) {
  if (currentSortColumn === colKey) {
    isSortAsc = !isSortAsc;
  } else {
    currentSortColumn = colKey;
    isSortAsc = true;
  }
  renderAlertsTable(allAlertsData);
}
