#!/bin/bash
# Production Deployment Script for Parsely AI

set -e

echo "🌿 Starting Parsely AI Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="parsely-ai"
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
check_requirements() {
    log_info "Checking deployment requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warn ".env file not found. Creating from template..."
        cp .env.production .env
        log_warn "Please update .env file with your production values before continuing."
        read -p "Press Enter to continue after updating .env file..."
    fi
    
    log_info "Requirements check passed ✅"
}

# Backup existing deployment
backup_existing() {
    if [ -d "data" ] || [ -d "logs" ]; then
        log_info "Creating backup of existing data..."
        mkdir -p "$BACKUP_DIR"
        
        if [ -d "data" ]; then
            cp -r data "$BACKUP_DIR/"
        fi
        
        if [ -d "logs" ]; then
            cp -r logs "$BACKUP_DIR/"
        fi
        
        log_info "Backup created at $BACKUP_DIR ✅"
    fi
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p data/policies
    mkdir -p data/embeddings
    mkdir -p logs
    mkdir -p ssl
    
    # Set proper permissions
    chmod 755 data/policies
    chmod 755 data/embeddings
    chmod 755 logs
    
    log_info "Directories created ✅"
}

# Build and deploy
deploy() {
    log_info "Building and deploying containers..."
    
    # Stop existing containers
    docker-compose down --remove-orphans
    
    # Build new images
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d
    
    log_info "Deployment completed ✅"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API health check passed ✅"
    else
        log_error "API health check failed ❌"
        log_info "Checking logs..."
        docker-compose logs llm-api
        exit 1
    fi
    
    # Check Nginx
    if curl -f http://localhost/health > /dev/null 2>&1; then
        log_info "Nginx health check passed ✅"
    else
        log_warn "Nginx health check failed (this might be expected if SSL is not configured)"
    fi
}

# Show deployment info
show_info() {
    log_info "Deployment Information:"
    echo "=========================="
    echo "🌐 API URL: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo "🔍 Health Check: http://localhost:8000/health"
    echo "📊 Metrics: http://localhost:8000/metrics"
    echo ""
    echo "🐳 Docker Services:"
    docker-compose ps
    echo ""
    echo "📋 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
    echo "🔄 To restart: docker-compose restart"
}

# Main deployment process
main() {
    log_info "Parsely AI - Production Deployment"
    echo "======================================================="
    
    check_requirements
    backup_existing
    setup_directories
    deploy
    health_check
    show_info
    
    log_info "🎉 Deployment completed successfully!"
    log_info "Your Parsely AI system is now running in production mode."
}

# Run main function
main "$@"