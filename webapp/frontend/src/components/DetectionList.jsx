import { useEffect, useRef, useState } from 'react'

export default function DetectionList({ detections }) {
  const [stableDetections, setStableDetections] = useState([])
  const prevStatusRef = useRef({})

  useEffect(() => {
    if (!detections) return

    // Napravi mapu trenutnih detekcija po statusu+imenu
    const currentMap = {}
    detections.forEach((det, i) => {
      const key = det.name || det.status
      currentMap[key] = det
    })

    // Usporedi s prethodnim stanjem
    const prev = prevStatusRef.current
    const changed = JSON.stringify(Object.keys(currentMap).sort()) !==
                    JSON.stringify(Object.keys(prev).sort())

    if (changed || stableDetections.length === 0 && detections.length > 0) {
      setStableDetections(detections)
      prevStatusRef.current = currentMap
    }

    // Ako nema detekcija, očisti
    if (detections.length === 0 && stableDetections.length > 0) {
      setStableDetections([])
      prevStatusRef.current = {}
    }
  }, [detections])

  return (
    <div className="detection-list">
      <p className="detection-list-title">Detekcije</p>
      {!stableDetections || stableDetections.length === 0 ? (
        <p className="detection-empty">Nema aktivnih detekcija</p>
      ) : (
        stableDetections.map((det, i) => (
          <div key={i} className={`detection-item ${det.status}`}>
            <span className="det-dot" />
            <div className="det-info">
              <span className="det-name">
                {det.status === 'known' && det.name}
                {det.status === 'unknown' && 'Nepoznata osoba'}
                {det.status === 'no_face' && 'Osoba'}
              </span>
            </div>
          </div>
        ))
      )}
    </div>
  )
}