import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function DetectionHistory() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    const fetch = () => {
      axios.get(`${API_URL}/history`)
        .then(res => setHistory([...res.data].reverse()))
        .catch(() => {})
    }
    fetch()
    const interval = setInterval(fetch, 2000)
    return () => clearInterval(interval)
  }, [])

  const formatTime = (iso) => {
    return new Date(iso).toLocaleTimeString('hr-HR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  return (
    <div className="history">
      <p className="section-title">Povijest detekcija</p>
      {history.length === 0 ? (
        <p className="empty-state">Nema zabilježenih detekcija</p>
      ) : (
        <div className="history-list">
          {history.map((entry, i) => (
            <div key={i} className={`history-item ${entry.status}`}>
              <span className="history-dot" />
              <div className="history-info">
                <span className="history-name">
                  {entry.status === 'known' && entry.name}
                  {entry.status === 'unknown' && 'Nepoznata osoba'}
                  {entry.status === 'no_face' && 'Osoba'}
                </span>
                <span className="history-score">
                  {entry.status === 'known' && `sličnost: ${entry.face_score?.toFixed(2)}`}
                  {entry.status === 'unknown' && `max: ${entry.face_score?.toFixed(2)}`}
                  {entry.status === 'no_face' && `conf: ${entry.confidence?.toFixed(2)}`}
                </span>
              </div>
              <span className="history-time">{formatTime(entry.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}