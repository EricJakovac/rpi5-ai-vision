import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function KnownPersons({ refreshKey }) {
  const [persons, setPersons] = useState([])
  const [deleting, setDeleting] = useState({})

  const fetchPersons = () => {
    axios.get(`${API_URL}/persons`)
      .then(res => setPersons(res.data.persons))
      .catch(() => {})
  }

  useEffect(() => {
    fetchPersons()
    const interval = setInterval(fetchPersons, 10000) 
    return () => clearInterval(interval)
  }, [refreshKey])

  useEffect(() => {
    const cropInterval = setInterval(() => {
      persons.forEach(p => fetchCrop(p.name))
    }, 5000)
    return () => clearInterval(cropInterval)
  }, [persons])

  const handleDelete = async (name) => {
    if (!window.confirm(`Jeste li sigurni da želite obrisati ${name} iz baze?`)) return
    setDeleting(prev => ({ ...prev, [name]: true }))
    try {
      await axios.delete(`${API_URL}/persons/${encodeURIComponent(name)}`)
      setPersons(prev => prev.filter(p => p.name !== name))
    } catch (err) {
      console.error('Greška pri brisanju:', err)
    } finally {
      setDeleting(prev => ({ ...prev, [name]: false }))
    }
  }

  const formatDate = (iso) => {
    if (!iso) return ''
    return new Date(iso).toLocaleDateString('hr-HR')
  }

  if (persons.length === 0) return null

  return (
    <div className="known-persons">
      <p className="section-title">Poznate osobe</p>
      <div className="persons-list">
        {persons.map((person, i) => (
          <div key={i} className="person-card">
            <div className="person-avatar">
              {person.name.charAt(0).toUpperCase()}
            </div>
            <div className="person-info">
              <span className="person-name">{person.name}</span>
              <span className="person-meta">
                {person.num_images} slika · {formatDate(person.registered)}
              </span>
            </div>
            <button
              className="person-delete-btn"
              onClick={() => handleDelete(person.name)}
              disabled={deleting[person.name]}
              title="Obriši osobu"
            >
              {deleting[person.name] ? '...' : '✕'}
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}