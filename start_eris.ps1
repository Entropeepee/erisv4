Write-Host "Starting Eris v4 Ecosystem..."

# 1. Start Ollama Server
Write-Host "Starting LLM Backend (Ollama: gpt-oss)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"

Start-Sleep -Seconds 3

# 2. Start FastAPI Server
Write-Host "Starting FRACTAL PDE FastAPI Server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\david\.gemini\antigravity\scratch\erisv4; uvicorn eris.server.app:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 3

# 3. Start React UI
Write-Host "Starting React Interface..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd C:\Users\david\.gemini\antigravity\scratch\erisv4\eris-ui; npm run dev"

Start-Sleep -Seconds 4
Start-Process "http://localhost:3000"

Write-Host "All systems launched! A browser window was opened at http://localhost:3000"
