function initCharts(summaryData) {
  const labels = summaryData.all_sentences.map((_, index) => `Sentence ${index + 1}`);
  const selectedIndices = new Set(summaryData.selected_indices);
  const scores = summaryData.all_scores;
  const backgroundColors = scores.map((_, index) =>
    selectedIndices.has(index)
      ? "rgba(59, 130, 246, 0.95)"
      : "rgba(148, 163, 184, 0.45)"
  );

  const scoreCtx = document.getElementById("scoreChart").getContext("2d");
  new Chart(scoreCtx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "TF-IDF score",
          data: scores,
          backgroundColor: backgroundColors,
          borderRadius: 12,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => `Score: ${ctx.parsed.y.toFixed(4)}` } },
      },
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { display: false } },
        y: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(148,163,184,0.12)" } },
      },
    },
  });

  const coverageCtx = document.getElementById("coverageChart").getContext("2d");
  const selectedCount = summaryData.selected_indices.length;
  const remainingCount = labels.length - selectedCount;

  new Chart(coverageCtx, {
    type: "doughnut",
    data: {
      labels: ["Included", "Skipped"],
      datasets: [
        {
          data: [selectedCount, remainingCount],
          backgroundColor: ["#22c55e", "#475569"],
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
      },
    },
  });
}
