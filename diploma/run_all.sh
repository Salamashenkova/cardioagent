#!/bin/bash
cd "$(dirname "$0")"  # Корень проекта

echo "🚀 GigaCardioAgent v2.1 — Полный запуск (WSL)"

WSL_IP="172.18.175.3"

# Backend FastAPI (порт 8000)
echo "🐳 Backend: $WSL_IP:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3

# Frontend Streamlit (порт 8501)
echo "🎨 Frontend: $WSL_IP:8501"
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "✅ Backend API: http://$WSL_IP:8000/docs"
echo "✅ Frontend UI: http://$WSL_IP:8501"
echo "🛑 Ctrl+C дважды для остановки"
echo ""

trap "kill \$BACKEND_PID \$FRONTEND_PID 2>/dev/null" INT
wait