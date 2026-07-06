export default function MetricsDashboard({ metrics }) {
  if (!metrics) return (
    <p className="metrics-loading">Čekam podatke...</p>
  )

  const ramPercent = ((metrics.ram_used_mb / metrics.ram_total_mb) * 100).toFixed(1)
  const tempClass = metrics.temperature_c > 75 ? 'danger' : metrics.temperature_c > 65 ? 'warning' : ''
  const cpuClass = metrics.cpu_percent > 90 ? 'danger' : metrics.cpu_percent > 75 ? 'warning' : ''

  return (
    <div>
      <p className="metrics-title">Performanse sustava</p>
      <div className="metrics-grid">
        <div className="metric-card">
          <span className="metric-label">FPS</span>
          <span className="metric-value">{metrics.fps}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Inference</span>
          <span className="metric-value small">
            {metrics.inference_ms}ms
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">CPU</span>
          <span className={`metric-value ${cpuClass}`}>{metrics.cpu_percent}%</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">RAM</span>
          <span className="metric-value">{ramPercent}%</span>
          <span className="metric-sub">{metrics.ram_used_mb}MB</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Temperatura</span>
          <span className={`metric-value ${tempClass}`}>
            {metrics.temperature_c}°C
          </span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Osobe</span>
          <span className="metric-value">{metrics.num_persons}</span>
        </div>
      </div>

      <div className="metrics-detail">
        <span>Detekcija: {metrics.detection_ms}ms</span>
        <span>Recognition: {metrics.recognition_ms}ms</span>
      </div>
    </div>
  )
}