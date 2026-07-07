import { useState, useEffect } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import VideoStream from './components/VideoStream'
import MetricsDashboard from './components/MetricsDashboard'
import ModelSelector from './components/ModelSelector'
import DetectionList from './components/DetectionList'
import PIRStatus from './components/PIRStatus'
import KnownPersons from './components/KnownPersons'
import ConfidenceMeter from './components/ConfidenceMeter'
import UnknownClusters from './components/UnknownClusters'
import './App.css'
import './Animations.css'

export default function App() {
  const { metrics, detections, connected, connecting } = useWebSocket()
  const [activeModel, setActiveModel] = useState(null)

  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <div className="app">
      <header className="app-header">
        <h1>RPi5 AI Vision</h1>
        <div className="header-right">
          <PIRStatus active={metrics?.pir_active} />
          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            {connecting && !connected
              ? <span className="status-spinner" />
              : <span className="status-dot" />
            }
            <span>
              {connected ? 'Spojeno' : connecting ? 'Spajam se...' : 'Odspojeno'}
            </span>
          </div>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      <main className="app-main">
        <div className="left-panel">
          <VideoStream connected={connected} />
          <div className="card">
            <DetectionList detections={detections} />
          </div>
          <div className="card">
            <ConfidenceMeter detections={detections} />
          </div>
        </div>

        <div className="right-panel">
          <div className="card">
            <ModelSelector
              onModelChange={setActiveModel}
              currentMetrics={metrics}
            />
          </div>
          <div className="card">
            <MetricsDashboard metrics={metrics} />
          </div>
          <div className="card">
            <KnownPersons />
          </div>
          <div className="card">
            <UnknownClusters />
          </div>
        </div>
      </main>
    </div>
  )
}