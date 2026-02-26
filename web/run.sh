#!/bin/bash

# Aviation RAG Web Interface - Startup Script

echo "🛩️  Aviation RAG Web Interface - Starting..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and set your API_KEY!"
    echo ""
fi

# Start the application
echo "Starting application..."
echo "Access the web interface at: http://localhost:8082"
echo ""
python main.py
