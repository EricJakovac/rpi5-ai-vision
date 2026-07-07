import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://192.168.1.234:8000/ws'

export function useWebSocket() {
  const [metrics, setMetrics] = useState(null)
  const [detections, setDetections] = useState([])
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(true)  
  const wsRef = useRef(null)
  const reconnectTimeout = useRef(null)

  const connect = useCallback(() => {
    setConnecting(true)  
    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setConnecting(false)  
        console.log('WebSocket spojen')
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        setMetrics(data.metrics)
        setDetections(data.detections || [])
      }

      ws.onclose = () => {
        setConnected(false)
        setConnecting(false)  
        console.log('WebSocket odspojoen, reconnect za 3s...')
        reconnectTimeout.current = setTimeout(() => {
          setConnecting(true)  
          connect()
        }, 3000)
      }

      ws.onerror = () => {
        setConnecting(false)
        ws.close()
      }
    } catch (error) {
      setConnecting(false)
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

  return { metrics, detections, connected, connecting }  
}