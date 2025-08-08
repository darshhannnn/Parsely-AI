@echo off
REM Production Deployment Script for Windows
REM Parsely AI

echo 🌿 Starting Parsely AI Production Deployment...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not available. Please install Docker Compose.
    pause
    exit /b 1
)

echo ✅ Docker requirements check passed

REM Check if .env file exists
if not exist .env (
    echo ⚠️ .env file not found. Creating from template...
    copy .env.production .env
    echo ⚠️ Please update .env file with your production values before continuing.
    pause
)

REM Create necessary directories
echo 📁 Setting up directories...
if not exist data\policies mkdir data\policies
if not exist data\embeddings mkdir data\embeddings
if not exist logs mkdir logs
if not exist ssl mkdir ssl

echo ✅ Directories created

REM Stop existing containers
echo 🛑 Stopping existing containers...
docker-compose down --remove-orphans

REM Build and start services
echo 🔨 Building and starting services...
docker-compose build --no-cache
docker-compose up -d

REM Wait for services to start
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Health check
echo 🏥 Performing health check...
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API health check failed
    echo 📋 Checking logs...
    docker-compose logs llm-api
    pause
    exit /b 1
)

echo ✅ API health check passed

REM Show deployment info
echo.
echo 🎉 Deployment completed successfully!
echo ================================
echo 🌐 API URL: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo 🔍 Health Check: http://localhost:8000/health
echo 📊 Metrics: http://localhost:8000/metrics
echo.
echo 🐳 Docker Services:
docker-compose ps
echo.
echo 📋 To view logs: docker-compose logs -f
echo 🛑 To stop: docker-compose down
echo 🔄 To restart: docker-compose restart
echo.
echo Your Parsely AI system is now running in production mode!

pause