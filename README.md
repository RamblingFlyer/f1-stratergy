# F1 Strategy Predictor

A machine learning and generative AI-powered system for predicting Formula 1 pit stop strategies, specifically focusing on undercut and overcut maneuvers.

## Overview

This project simulates mid-race pit stop strategies in Formula 1 and predicts the success of an undercut or overcut maneuver using machine learning. Enhanced with Generative AI and powered by real-time APIs, it acts as a virtual race engineer, providing smart, data-backed decisions and human-like strategy explanations through a Streamlit interface.

## Key Features

### ML-Based Strategy Prediction
- Binary classification (Success/Fail) for undercut/overcut based on tire delta, pace drop-off, track gap, tire degradation curve, and rival pit window
- Output: Success probability + confidence score
- API: ML models are served via a FastAPI backend, which Streamlit queries in real-time using REST calls

### Scenario Simulation Engine
- Simulates multiple strategic possibilities:
  - Pit now vs. in 1–3 laps
  - Different tire compounds
  - Varying track/weather conditions
  - Safety car appearance simulations
- Backend: Logic connected via REST APIs for scalable deployment
- Data Source (Optional): Uses FastF1 API to optionally pull real-world race data like lap times, tire life, and sector deltas

### Generative AI Integration
- LLM-Powered Strategy Assistant using OpenAI GPT-4 API
- Alternate Timeline Generator for "what-if" scenarios
- Post-Race Debrief Generator
- Natural Language Input + Output

### Streamlit Interface
- Clean, interactive UI with inputs for live race metrics
- Real-time ML prediction visualization
- Strategy comparison charts
- AI-powered chat sidebar for Q&A

## Tech Stack & APIs
- ML Model: scikit-learn / XGBoost (served via FastAPI)
- GenAI Strategy Bot: OpenAI GPT-4 API
- Real Race Data: FastF1 Python API (optional)
- UI: Streamlit
- Communication: requests, openai, REST API calls

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the FastAPI backend
```bash
cd backend
uvicorn main:app --reload
```

### Start the Streamlit frontend
```bash
cd frontend
streamlit run app.py
```

## Project Structure

```
├── backend/                # FastAPI backend
│   ├── main.py            # FastAPI app
│   ├── models/            # ML models
│   │   ├── predictor.py   # ML prediction logic
│   │   └── train.py       # Model training script
│   └── simulation/        # Simulation engine
│       └── engine.py      # Race simulation logic
├── frontend/              # Streamlit frontend
│   └── app.py             # Streamlit app
├── data/                  # Data directory
│   └── sample_data.csv    # Sample data for testing
├── utils/                 # Utility functions
│   ├── fastf1_helper.py   # FastF1 API helper
│   └── openai_helper.py   # OpenAI API helper
└── requirements.txt       # Project dependencies
```