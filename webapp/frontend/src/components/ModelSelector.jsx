import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

const SHORT_NAMES = {
  'yolov8n_int8.tflite':  'YOLOv8n INT8 ⭐',
  'yolov8n_fp32.tflite':  'YOLOv8n FP32',
  'yolov8s_int8.tflite':  'YOLOv8s INT8',
  'yolov8s_fp32.tflite':  'YOLOv8s FP32',
  'yolov10n_int8.tflite': 'YOLOv10n INT8',
  'yolo11n_int8.tflite':  'YOLOv11n INT8',
}

export default function ModelSelector({ onModelChange }) {
  const [models, setModels] = useState([])
  const [current, setCurrent] = useState(null)
  const [loading, setLoading] = useState(false)
  const [updating, setUpdating] = useState(false)

  useEffect(() => {
    axios.get(`${API_URL}/models`).then(res => {
      const sorted = [...res.data.available].sort(
        (a, b) => b.map_score - a.map_score
      )
      setModels(sorted)
      setCurrent(res.data.current)
    }).catch(console.error)
  }, [])

  const handleChange = async (filename) => {
    if (filename === current?.filename || loading) return
    setLoading(true)

    // Fade out
    setUpdating(true)

    try {
      const res = await axios.post(`${API_URL}/models/switch`, { filename })
      if (res.data.success) {
        const selected = models.find(m => m.filename === filename)

        // Čekaj fade out pa ažuriraj
        setTimeout(() => {
          setCurrent(selected)
          onModelChange(res.data.active_model)
          // Fade in
          setUpdating(false)
        }, 250)
      } else {
        setUpdating(false)
      }
    } catch (err) {
      console.error('Greška pri promjeni modela:', err)
      setUpdating(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="model-selector">
      <label>Detekcijski model</label>
      <select
        onChange={e => handleChange(e.target.value)}
        value={current?.filename || ''}
        disabled={loading}
      >
        {models.map(m => (
          <option key={m.filename} value={m.filename}>
            {SHORT_NAMES[m.filename] || m.name}
          </option>
        ))}
      </select>

      {loading && <p className="model-switching">Mijenjam model...</p>}

      {current && (
        <div className={`model-detail ${updating ? 'updating' : ''}`}>
          <div className="model-stat">
            <span className="model-stat-label">mAP@0.5</span>
            <span className="model-stat-value">{current.map_score}</span>
          </div>
          <div className="model-stat">
            <span className="model-stat-label">Benchmark FPS</span>
            <span className="model-stat-value">{current.benchmark_fps}</span>
          </div>
          <div className="model-stat">
            <span className="model-stat-label">Format</span>
            <span className="model-stat-value" style={{fontSize:'0.8rem'}}>
              {current.format?.toUpperCase()} {current.quantization?.toUpperCase()}
            </span>
          </div>
          <div className="model-stat">
            <span className="model-stat-label">Veličina</span>
            <span className="model-stat-value">{current.size_mb} MB</span>
          </div>
        </div>
      )}
    </div>
  )
}