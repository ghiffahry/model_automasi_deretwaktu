#!/bin/bash

echo "======================================"
echo "SURADATA Backend Startup Script"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    exit 1
fi

echo "Python version:"
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying critical packages..."
python3 -c "import fastapi; print('✓ FastAPI:', fastapi.__version__)" || echo "✗ FastAPI not installed"
python3 -c "import uvicorn; print('✓ Uvicorn installed')" || echo "✗ Uvicorn not installed"
python3 -c "from config import Config; print('✓ Config validated')" || echo "✗ Config error"

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo ""
    echo "WARNING: Port 8000 is already in use"
    echo "Kill the process or use a different port"
    exit 1
fi

# Start server
echo ""
echo "======================================"
echo "Starting SURADATA API Server..."
echo "======================================"
echo "API will be available at: http://localhost:8000"
echo "Documentation: http://localhost:8000/docs"
echo "Press CTRL+C to stop the server"
echo "======================================"
echo ""

uvicorn api:app --reload --host 0.0.0.0 --port 8000