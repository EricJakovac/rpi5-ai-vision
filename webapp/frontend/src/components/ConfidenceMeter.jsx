import { useEffect, useState } from 'react'

export default function ConfidenceMeter({ detections }) {
  const [displayed, setDisplayed] = useState([])

  useEffect(() => {
    if (!detections || detections.length === 0) {
      setDisplayed([])
      return
    }
    setDisplayed(detections)
  }, [detections])

  if (displayed.length === 0) return null

  return (
    <div className="confidence-meter">
      <p className="section-title">Pouzdanost detekcije</p>
      {displayed.map((det, i) => {
        const isKnown = det.status === 'known'
        const isUnknown = det.status === 'unknown'
        const isNoFace = det.status === 'no_face'

        const score = isKnown ? det.face_score :
                      isUnknown ? det.face_score :
                      det.confidence

        const percent = Math.round(Math.max(0, score) * 100)

        const color = isKnown ? 'var(--accent)' :
                      isUnknown ? 'var(--accent-red)' :
                      'var(--accent-blue)'

        const label = isKnown ? det.name :
                      isUnknown ? 'Nepoznata osoba' :
                      'Osoba'

        return (
          <div key={i} className="conf-item">
            <div className="conf-header">
              <span className="conf-label">{label}</span>
              <span className="conf-percent" style={{ color }}>
                {isNoFace ? `${percent}%` : `${percent}%`}
              </span>
            </div>
            <div className="conf-bar-bg">
              <div
                className="conf-bar-fill"
                style={{
                  width: `${percent}%`,
                  background: color,
                  transition: 'width 0.4s ease'
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}