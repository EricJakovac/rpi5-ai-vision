import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://192.168.1.234:8000/ws'

export function useWebSocket() {
  const [metrics, setMetrics] = useState(null)
  const [detections, setDetections] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectTimeout = useRef(null)

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        console.log('WebSocket spojen')
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        setMetrics(data.metrics)
        setDetections(data.detections || [])
      }

      ws.onclose = () => {
        setConnected(false)
        console.log('WebSocket odspojen, reconnect za 3s...')
        reconnectTimeout.current = setTimeout(connect, 3000)
      }

      ws.onerror = (error) => {
        console.error('WebSocket greška:', error)
        ws.close()
      }
    } catch (error) {
      console.error('WebSocket connection error:', error)
      reconnectTimeout.current = setTimeout(connect, 3000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current)
    }
  }, [connect])

  return { metrics, detections, connected }
}