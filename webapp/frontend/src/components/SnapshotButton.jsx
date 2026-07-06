import { useState } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function SnapshotButton() {
  const [loading, setLoading] = useState(false)
  const [lastTime, setLastTime] = useState(null)

  const takeSnapshot = async () => {
    setLoading(true)
    try {
      const res = await axios.get(`${API_URL}/snapshot`)
      setLastTime(new Date(res.data.timestamp).toLocaleTimeString('hr-HR'))

      const link = document.createElement('a')
      link.href = res.data.image
      link.download = `snapshot_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.jpg`
      link.click()
    } catch (err) {
      console.error('Snapshot greška:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="snapshot">
      <button
        className="snapshot-btn"
        onClick={takeSnapshot}
        disabled={loading}
      >
        {loading ? '⏳ Snimam...' : '📸 Spremi snapshot'}
      </button>
      {lastTime && (
        <span className="snapshot-time">Zadnji: {lastTime}</span>
      )}
    </div>
  )
}