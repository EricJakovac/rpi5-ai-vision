export default function PIRStatus({ active }) {
  return (
    <div className="pir-status">
      <span className={`pir-dot ${active ? 'active' : ''}`} />
      <span>{active ? 'PIR aktivan' : 'Čekam pokret'}</span>
    </div>
  )
}