#!/bin/bash
echo "================================================"
echo "  ChurnBank - Customer Churn Prediction System"
echo "================================================"
echo ""
echo "Step 1: Installing dependencies..."
pip install -r requirements.txt
echo ""
echo "Step 2: Training ML models..."
python generate_and_train.py
echo ""
echo "Step 3: Starting web application..."
echo ""
echo "Open your browser at: http://127.0.0.1:5000"
echo "Press CTRL+C to stop the server"
echo ""
python app.py
