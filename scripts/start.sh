#!/bin/bash
echo "🚀 Pokretanje RPi5 AI Vision..."

# Aktiviraj venv
source ~/Desktop/rpi5-ai-vision/venv/bin/activate

# Pokreni backend u pozadini
cd ~/Desktop/rpi5-ai-vision/webapp/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "✅ Backend pokrenut (PID: $BACKEND_PID)"

# Čekaj da se backend stvarno pokrene
echo "⏳ Čekam inicijalizaciju..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend spreman!"
        break
    fi
    sleep 1
    echo "   Čekam... ($i/30)"
done

# Provjeri je li backend uopće pokrenut
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ Backend nije pokrenut! Provjeri greške iznad."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Pokreni cloudflared i izvuci URL
echo "🌐 Pokretanje Cloudflare Tunnel..."
cloudflared tunnel --url http://localhost:8000 2>&1 | while IFS= read -r line; do
    echo "$line"
    if echo "$line" | grep -q "trycloudflare.com"; then
        URL=$(echo "$line" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com')
        if [ -n "$URL" ]; then
            echo ""
            echo "╔══════════════════════════════════════════════════════╗"
            echo "║  🌍 PRODUKCIJSKI URL:                                ║"
            echo "║  $URL"
            echo "║                                                      ║"
            echo "║  Ažuriraj na Vercel:                                 ║"
            echo "║  VITE_API_URL=$URL"
            echo "║  VITE_WS_URL=${URL/https/wss}/ws"
            echo "║  VITE_STREAM_URL=$URL/stream"
            echo "╚══════════════════════════════════════════════════════╝"
            echo ""
        fi
    fi
done

# Cleanup
trap "echo '🛑 Gašenje...'; kill $BACKEND_PID 2>/dev/null; exit" INT TERM