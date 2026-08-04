import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function UnknownClusters({ onPersonAdded }) {
  const [clusters, setClusters] = useState([])
  const [stats, setStats] = useState(null)
  const [resetting, setResetting] = useState(false)
  const [crops, setCrops] = useState({})
  const [addingName, setAddingName] = useState({})
  const [nameInputs, setNameInputs] = useState({})
  const [dismissing, setDismissing] = useState({})
  const [errors, setErrors] = useState({})

  const fetchClusters = () => {
    axios.get(`${API_URL}/clusters`)
      .then(res => {
        setClusters(res.data.clusters)
        setStats(res.data.stats)
        res.data.clusters.forEach(c => {
          if (crops[c.cluster_id] === undefined) {
            fetchCrop(c.cluster_id)
          }
        })
      })
      .catch(() => {})
  }

  useEffect(() => {
    fetchClusters()
    const interval = setInterval(fetchClusters, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchCrop = async (clusterId) => {
    try {
      const res = await axios.get(`${API_URL}/clusters/${clusterId}/crop`)
      if (res.data.success && res.data.crop) {
        setCrops(prev => ({ ...prev, [clusterId]: res.data.crop }))
      } else {
        setCrops(prev => ({ ...prev, [clusterId]: null }))
      }
    } catch {
      setCrops(prev => ({ ...prev, [clusterId]: null }))
    }
  }

  const handleReset = async () => {
    if (!window.confirm('Jeste li sigurni da želite resetirati sve klastere nepoznatih osoba?')) return
    setResetting(true)
    try {
      await axios.post(`${API_URL}/clusters/reset`)
      setClusters([])
      setStats(null)
      setCrops({})
    } catch (err) {
      console.error('Reset greška:', err)
    } finally {
      setResetting(false)
    }
  }

  const handleAddToDatabase = async (clusterId) => {
    const name = nameInputs[clusterId]?.trim()
    if (!name) {
      setErrors(prev => ({ ...prev, [clusterId]: 'Unesite ime!' }))
      return
    }
    setErrors(prev => ({ ...prev, [clusterId]: null }))
    setAddingName(prev => ({ ...prev, [clusterId]: true }))
    try {
      const res = await axios.post(
        `${API_URL}/clusters/${clusterId}/add-to-database`,
        { name }
      )
      if (res.data.success) {
        await axios.post(`${API_URL}/clusters/${clusterId}/dismiss`)
        setClusters(prev => prev.filter(c => c.cluster_id !== clusterId))
        setCrops(prev => { const n = { ...prev }; delete n[clusterId]; return n })
        setNameInputs(prev => { const n = { ...prev }; delete n[clusterId]; return n })
        if (onPersonAdded) onPersonAdded()
      }
    } catch (err) {
      console.error('Greška pri dodavanju:', err)
    } finally {
      setAddingName(prev => ({ ...prev, [clusterId]: false }))
    }
  }

  const handleDismiss = async (clusterId) => {
    setDismissing(prev => ({ ...prev, [clusterId]: true }))
    try {
      await axios.post(`${API_URL}/clusters/${clusterId}/dismiss`)
      setClusters(prev => prev.filter(c => c.cluster_id !== clusterId))
      setCrops(prev => { const n = { ...prev }; delete n[clusterId]; return n })
    } catch (err) {
      console.error('Greška pri odbijanju:', err)
    } finally {
      setDismissing(prev => ({ ...prev, [clusterId]: false }))
    }
  }

  const formatTime = (iso) => {
    if (!iso) return ''
    return new Date(iso).toLocaleString('hr-HR', {
      day: '2-digit', month: '2-digit',
      hour: '2-digit', minute: '2-digit'
    })
  }

  return (
    <div className="unknown-clusters">
      <div className="clusters-header">
        <p className="section-title">Nepoznate osobe</p>
        {stats && stats.total_embeddings > 0 && (
          <button className="reset-btn" onClick={handleReset} disabled={resetting}>
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

              {/* Avatar / Crop u malom krugu */}
              <div className="cluster-avatar-wrap">
                {crops[cluster.cluster_id] ? (
                  <img
                    src={`data:image/jpeg;base64,${crops[cluster.cluster_id]}`}
                    alt={`Osoba #${i + 1}`}
                    className="cluster-avatar-img"
                  />
                ) : (
                  <div className="cluster-avatar">{i + 1}</div>
                )}
              </div>

              {/* Info + akcije */}
              <div className="cluster-info">
                <span className="cluster-name">Nepoznata osoba #{i + 1}</span>
                <span className="cluster-meta">Viđena {cluster.count}×</span>
                <span className="cluster-time">
                  {formatTime(cluster.first_seen)} – {formatTime(cluster.last_seen)}
                </span>

                <div className="cluster-actions">
                  <div className="cluster-input-wrap">
                    <input
                      type="text"
                      placeholder="Unesite ime nepoznate osobe..."
                      value={nameInputs[cluster.cluster_id] || ''}
                      onChange={e => {
                        setNameInputs(prev => ({
                          ...prev, [cluster.cluster_id]: e.target.value
                        }))
                        setErrors(prev => ({ ...prev, [cluster.cluster_id]: null }))
                      }}
                      onKeyDown={e => e.key === 'Enter' && handleAddToDatabase(cluster.cluster_id)}
                      className={`cluster-name-input ${errors[cluster.cluster_id] ? 'input-error' : ''}`}
                    />
                    {errors[cluster.cluster_id] && (
                      <span className="cluster-error">{errors[cluster.cluster_id]}</span>
                    )}
                  </div>
                  <button
                    className="cluster-add-btn"
                    onClick={() => handleAddToDatabase(cluster.cluster_id)}
                    disabled={addingName[cluster.cluster_id]}
                    title="Potvrdi"
                  >
                    {addingName[cluster.cluster_id] ? '...' : '✓'}
                  </button>
                  <button
                    className="cluster-dismiss-btn"
                    onClick={() => handleDismiss(cluster.cluster_id)}
                    disabled={dismissing[cluster.cluster_id]}
                    title="Odbij"
                  >
                    {dismissing[cluster.cluster_id] ? '...' : '✕'}
                  </button>
                </div>
              </div>

            </div>
          ))}
        </div>
      )}
    </div>
  )
}