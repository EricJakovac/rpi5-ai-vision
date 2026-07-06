const STREAM_URL = import.meta.env.VITE_STREAM_URL || 'http://192.168.1.234:8000/stream'

export default function VideoStream({ connected }) {
  return (
    <div className="video-wrapper">
      {connected ? (
        <img
          src={STREAM_URL}
          alt="Live stream"
          className="video-stream"
        />
      ) : (
        <div className="video-placeholder">
          <span>📷</span>
          <p>Spajam se na kameru...</p>
        </div>
      )}
    </div>
  )
}