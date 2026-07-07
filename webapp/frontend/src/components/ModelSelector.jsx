import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || 'http://192.168.1.234:8000'

// Kratki nazivi za dropdown
const SHORT_NAMES = {
  'yolov8n_int8.tflite':  'YOLOv8n INT8 ⭐',
  'yolov8n_fp32.tflite':  'YOLOv8n FP32',
  'yolov8s_int8.tflite':  'YOLOv8s INT8',
  'yolov10n_int8.tflite': 'YOLOv10n INT8',
  'yolo11n_int8.tflite':  'YOLOv11n INT8',
}

export default function ModelSelector({ onModelChange, currentMetrics }) {
  const [models, setModels] = useState([])
  const [current, setCurrent] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    axios.get(`${API_URL}/models`).then(res => {
      // Sortiraj po mAP od najboljeg prema najlošijem
      const sorted = [...res.data.available].sort(
        (a, b) => b.map_score - a.map_score
      )
      setModels(sorted)
      setCurrent(res.data.current)
    }).catch(console.error)
  }, [])

  const handleChange = async (filename) => {
    setLoading(true)
    try {
      const res = await axios.post(`${API_URL}/models/switch`, { filename })
      if (res.data.success) {
        const selected = models.find(m => m.filename === filename)
        setCurrent(selected)
        onModelChange(res.data.active_model)
      }
    } catch (err) {
      console.error('Greška pri promjeni modela:', err)
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
        <div className="model-detail">
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
            <span className="model-stat-value" style={{fontSize:'0.9rem'}}>
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