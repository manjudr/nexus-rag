#!/bin/bash
# Quick test commands for NexusRAG system

echo "🚀 Quick NexusRAG Tests"
echo "======================="

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "1. System Status:"
python main_app.py --status

echo ""
echo "2. Load PDF with LangExtract:"
python main_app.py --load-small

echo ""
echo "3. PDF Educational Query:"
python main_app.py --query "What are the main learning objectives?"

echo ""
echo "4. Video Query (should route to video agent):"
python main_app.py --query "Show me a video tutorial"

echo ""
echo "5. General Query (should route to general agent):"
python main_app.py --query "What is machine learning?"

echo ""
echo "6. Complex Educational Query:"
python main_app.py --query "I want to learn data science step by step"

echo ""
echo "✅ Quick tests complete!"
