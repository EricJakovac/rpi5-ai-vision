import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

export default function KnownPersons() {
  const [persons, setPersons] = useState([])

  useEffect(() => {
    axios.get(`${API_URL}/persons`)
      .then(res => setPersons(res.data.persons))
      .catch(() => {})
  }, [])

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
          </div>
        ))}
      </div>
    </div>
  )
}