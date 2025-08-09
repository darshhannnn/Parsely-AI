@echo off
echo 🚂 Railway Deployment Script
echo ============================

echo Installing Railway CLI...
npm install -g @railway/cli

echo.
echo Logging into Railway...
railway login

echo.
echo Creating new Railway project...
railway new

echo.
echo Setting environment variables...
railway variables set GOOGLE_API_KEY=AIzaSyBtgCpWhnVVRknJvvKbpSJ-MzGCniEXLuM
railway variables set HACKATHON_API_TOKEN=8e6a11e26a0e51d768ce7fb55743017cb25ee7c6891e15c4ab2f1bf971bf9d63
railway variables set LLM_MODEL=gemini-1.5-flash
railway variables set PYTHONPATH=/app

echo.
echo Deploying to Railway...
railway up

echo.
echo ✅ Deployment complete!
echo Your API will be available at: https://your-app.railway.app/hackrx/run
echo.
pause