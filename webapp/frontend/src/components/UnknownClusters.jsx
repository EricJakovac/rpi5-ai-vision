import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function UnknownClusters() {
  const [clusters, setClusters] = useState([])
  const [stats, setStats] = useState(null)
  const [resetting, setResetting] = useState(false)

  useEffect(() => {
    const fetch = () => {
      axios.get(`${API_URL}/clusters`)
        .then(res => {
          setClusters(res.data.clusters)
          setStats(res.data.stats)
        })
        .catch(() => {})
    }
    fetch()
    const interval = setInterval(fetch, 5000)
    return () => clearInterval(interval)
  }, [])

  const handleReset = async () => {
    if (!window.confirm('Resetirati sve klastere nepoznatih osoba?')) return
    setResetting(true)
    try {
      await axios.post(`${API_URL}/clusters/reset`)
      setClusters([])
      setStats(null)
    } catch (err) {
      console.error('Reset greška:', err)
    } finally {
      setResetting(false)
    }
  }

  const formatTime = (iso) => {
    if (!iso) return ''
    return new Date(iso).toLocaleString('hr-HR', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="unknown-clusters">
      <div className="clusters-header">
        <p className="section-title">Nepoznate osobe</p>
        {stats && stats.total_embeddings > 0 && (
          <button
            className="reset-btn"
            onClick={handleReset}
            disabled={resetting}
          >
            {resetting ? '...' : '↺'}
          </button>
        )}
      </div>

      {stats && (
        <div className="clusters-stats">
          <span>{stats.total_embeddings} uzoraka</span>
          <span>{stats.num_clusters} grupa</span>
          <span>{stats.num_outliers} outliera</span>
        </div>
      )}

      {clusters.length === 0 ? (
        <p className="empty-state">Nema grupiranih nepoznatih osoba</p>
      ) : (
        <div className="clusters-list">
          {clusters.map((cluster, i) => (
            <div key={cluster.cluster_id} className="cluster-item">
              <div className="cluster-avatar">
                {i + 1}
              </div>
              <div className="cluster-info">
                <span className="cluster-name">
                  Nepoznata osoba #{i + 1}
                </span>
                <span className="cluster-meta">
                  Viđena {cluster.count}×
                </span>
                <span className="cluster-time">
                  {formatTime(cluster.first_seen)} – {formatTime(cluster.last_seen)}
                </span>
              </div>
              <span className="cluster-badge">{cluster.count}×</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}