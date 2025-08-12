@echo off
echo Starting Parsely AI Hackathon Server with Ngrok...
echo.

REM Set environment variables
set HACKATHON_API_TOKEN=hackrx_2024_parsely_ai_token
set GOOGLE_API_KEY=%GOOGLE_API_KEY%
set LLM_MODEL=gemini-2.0-flash

echo Starting FastAPI server...
start "Parsely AI Server" python src/api/hackathon_main.py

echo Waiting for server to start...
timeout /t 5 /nobreak > nul

echo Starting ngrok tunnel...
ngrok http 8000

pause