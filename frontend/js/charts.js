/**
 * Chart.js Integration for Time-Series Demand Forecasting.
 * Renders historical actual demand, forecast prediction line,
 * uncertainty confidence interval band, and baseline comparison.
 */

let demandChartInstance = null;

function renderDemandForecastChart(historicalData, forecastData, baselineData = null) {
  const ctx = document.getElementById('demandForecastChart');
  if (!ctx) return;

  if (demandChartInstance) {
    demandChartInstance.destroy();
  }

  // Extract dates and values
  const histDates = historicalData.map(d => d.date);
  const histSales = historicalData.map(d => d.quantity_sold);

  const forecastDates = forecastData.map(d => d.date);
  const forecastPreds = forecastData.map(d => d.predicted_quantity);
  const lowerBounds = forecastData.map(d => d.lower_bound);
  const upperBounds = forecastData.map(d => d.upper_bound);

  const allLabels = [...histDates, ...forecastDates];

  // Align datasets across common timeline
  const histAligned = [...histSales, ...new Array(forecastDates.length).fill(null)];

  // Forecast connects to the last historical point
  const lastHistVal = histSales[histSales.length - 1];
  const forecastAligned = [
    ...new Array(histDates.length - 1).fill(null),
    lastHistVal,
    ...forecastPreds
  ];

  const lowerAligned = [
    ...new Array(histDates.length - 1).fill(null),
    lastHistVal,
    ...lowerBounds
  ];

  const upperAligned = [
    ...new Array(histDates.length - 1).fill(null),
    lastHistVal,
    ...upperBounds
  ];

  const datasets = [
    {
      label: 'Historical Actual Sales',
      data: histAligned,
      borderColor: '#94a3b8',
      backgroundColor: 'rgba(148, 163, 184, 0.1)',
      borderWidth: 2,
      pointRadius: 1,
      pointHoverRadius: 4,
      tension: 0.2,
      order: 3
    },
    {
      label: 'Prophet Demand Forecast',
      data: forecastAligned,
      borderColor: '#6366f1',
      backgroundColor: 'rgba(99, 102, 241, 0.2)',
      borderWidth: 3,
      borderDash: [0, 0],
      pointRadius: 2,
      pointHoverRadius: 5,
      tension: 0.25,
      order: 1
    },
    {
      label: 'Confidence Upper (95%)',
      data: upperAligned,
      borderColor: 'rgba(99, 102, 241, 0.2)',
      backgroundColor: 'rgba(99, 102, 241, 0.12)',
      borderWidth: 1,
      pointRadius: 0,
      fill: '+1', // fill down to lower bound
      tension: 0.25,
      order: 2
    },
    {
      label: 'Confidence Lower (95%)',
      data: lowerAligned,
      borderColor: 'rgba(99, 102, 241, 0.2)',
      backgroundColor: 'transparent',
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      tension: 0.25,
      order: 2
    }
  ];

  // Optional Baseline Linear Regression overlay
  if (baselineData && baselineData.length > 0) {
    const basePreds = baselineData.map(d => d.predicted_quantity);
    const baseAligned = [
      ...new Array(histDates.length - 1).fill(null),
      lastHistVal,
      ...basePreds
    ];
    datasets.push({
      label: 'Baseline Linear Reg',
      data: baseAligned,
      borderColor: '#f59e0b',
      borderWidth: 1.8,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
      tension: 0.1,
      order: 4
    });
  }

  demandChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#1a2c5b',
            font: { family: 'Inter', size: 12, weight: '600' },
            usePointStyle: true,
            boxWidth: 8
          }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#ffffff',
          bodyColor: '#f1f5f9',
          borderColor: 'rgba(255, 255, 255, 0.15)',
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 40, 80, 0.07)' },
          ticks: {
            color: '#475569',
            maxTicksLimit: 12,
            font: { family: 'Inter', size: 11, weight: '500' }
          }
        },
        y: {
          grid: { color: 'rgba(30, 40, 80, 0.07)' },
          ticks: {
            color: '#475569',
            font: { family: 'Inter', size: 11, weight: '500' }
          },
          title: {
            display: true,
            text: 'Daily Quantity Sold (Units)',
            color: '#1a2c5b',
            font: { family: 'Inter', size: 12, weight: '600' }
          },
          min: 0
        }
      }
    }
  });
}
