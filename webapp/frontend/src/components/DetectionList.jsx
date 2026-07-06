export default function DetectionList({ detections }) {
  return (
    <div className="detection-list">
      <p className="detection-list-title">Detekcije</p>
      {!detections || detections.length === 0 ? (
        <p className="detection-empty">Nema aktivnih detekcija</p>
      ) : (
        detections.map((det, i) => (
          <div key={i} className={`detection-item ${det.status}`}>
            <span className="det-dot" />
            <div className="det-info">
              <span className="det-name">
                {det.status === 'known' && det.name}
                {det.status === 'unknown' && 'Nepoznata osoba'}
                {det.status === 'no_face' && 'Osoba'}
              </span>
              <span className="det-score">
                {det.status === 'known' && `sličnost: ${det.face_score.toFixed(2)}`}
                {det.status === 'unknown' && `max: ${det.face_score.toFixed(2)}`}
                {det.status === 'no_face' && `conf: ${det.confidence.toFixed(2)}`}
              </span>
            </div>
          </div>
        ))
      )}
    </div>
  )
}