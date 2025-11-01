#!/bin/bash
# Momento Enhanced Memory - Quick Start Script

set -e

echo "======================================"
echo "Momento Enhanced Memory - Quick Start"
echo "======================================"
echo ""

# Check if Neo4j is running
echo "1. Checking Neo4j database..."
if docker ps | grep -q momento-neo4j; then
    echo "   ✓ Neo4j is already running"
else
    echo "   Starting Neo4j..."
    cd infra
    docker-compose up -d
    cd ..
    echo "   ✓ Neo4j started"
    echo "   Waiting for Neo4j to be ready..."
    sleep 10
fi

# Check Python version
echo ""
echo "2. Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ✗ Python 3 not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv .venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment exists"
fi

# Activate virtual environment
echo ""
echo "3. Installing dependencies..."
source .venv/bin/activate

# Install package
pip install -e . > /dev/null 2>&1
echo "   ✓ Momento installed"

# Run examples
echo ""
echo "4. Running basic examples..."
echo ""
python examples/basic_usage.py

echo ""
echo "======================================"
echo "Quick Start Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. View the graph in Neo4j Browser:"
echo "   http://localhost:7474"
echo "   Username: neo4j"
echo "   Password: neo4j_password"
echo ""
echo "2. Start the MCP server:"
echo "   source .venv/bin/activate"
echo "   momento --db-uri bolt://localhost:7687 --username neo4j --password neo4j_password"
echo ""
echo "3. Configure Claude Desktop:"
echo "   See README.md for configuration"
echo ""
echo "4. Read the documentation:"
echo "   - README.md - Overview and setup"
echo "   - DESIGN.md - Architecture and design details"
echo ""

