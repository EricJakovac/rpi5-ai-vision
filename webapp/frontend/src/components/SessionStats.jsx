import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function SessionStats() {
  const [session, setSession] = useState(null)

  useEffect(() => {
    const fetch = () => {
      axios.get(`${API_URL}/session`)
        .then(res => setSession(res.data))
        .catch(() => {})
    }
    fetch()
    const interval = setInterval(fetch, 3000)
    return () => clearInterval(interval)
  }, [])

  if (!session) return null

  return (
    <div className="session">
      <p className="section-title">Statistike sesije</p>
      <div className="session-grid">
        <div className="session-stat">
          <span className="session-label">Uptime</span>
          <span className="session-value">{session.uptime}</span>
        </div>
        <div className="session-stat">
          <span className="session-label">Ukupno</span>
          <span className="session-value">{session.total_detections}</span>
        </div>
        <div className="session-stat">
          <span className="session-label">Poznate</span>
          <span className="session-value known">{session.known_detections}</span>
        </div>
        <div className="session-stat">
          <span className="session-label">Nepoznate</span>
          <span className="session-value unknown">{session.unknown_detections}</span>
        </div>
      </div>

      {Object.keys(session.person_counts).length > 0 && (
        <div className="person-counts">
          {Object.entries(session.person_counts)
            .sort((a, b) => b[1] - a[1])
            .map(([name, count]) => (
              <div key={name} className="person-count-item">
                <span className="person-count-dot" />
                <span className="person-count-name">{name}</span>
                <span className="person-count-badge">{count}×</span>
              </div>
            ))}
        </div>
      )}
    </div>
  )
}