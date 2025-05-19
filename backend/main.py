from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
import json

# Import our custom modules
from models.predictor import StrategyPredictor
from models.strategy_advisor import StrategyAdvisor
from simulation.engine import RaceSimulator

# Initialize FastAPI app
app = FastAPI(
    title="F1 Strategy Predictor API",
    description="API for predicting F1 pit stop strategies and race simulations",
    version="1.0.0"
)

# Initialize our models
strategy_predictor = StrategyPredictor()
strategy_advisor = StrategyAdvisor()
race_simulator = RaceSimulator()

# Define request models
class UndercutRequest(BaseModel):
    tire_delta: float  # Difference in tire age (laps)
    pace_dropoff: float  # Pace drop-off per lap (seconds)
    track_gap: float  # Gap between cars (seconds)
    tire_deg_curve: float  # Tire degradation curve factor
    rival_pit_window: int  # Expected laps until rival pits
    
class OvercutRequest(BaseModel):
    tire_delta: float
    pace_dropoff: float
    track_gap: float
    tire_deg_curve: float
    rival_pit_window: int

class SimulationRequest(BaseModel):
    current_lap: int
    total_laps: int
    current_position: int
    gap_ahead: float  # Gap to car ahead (seconds)
    gap_behind: float  # Gap to car behind (seconds)
    current_tire_age: int  # Current tire age (laps)
    current_compound: str  # Current tire compound (e.g., "soft", "medium", "hard")
    weather_condition: str  # Current weather (e.g., "dry", "wet", "mixed")
    pit_scenarios: List[Dict[str, Any]]  # List of pit scenarios to simulate

# Define response models
class PredictionResponse(BaseModel):
    success_probability: float
    confidence_score: float
    recommended_action: str
    factors: Dict[str, float]  # Importance of each factor in the prediction

class StrategyAdviceResponse(BaseModel):
    recommended_strategy: str
    success_probability: float
    key_factors: Dict[str, float]

class SimulationResponse(BaseModel):
    scenarios: List[Dict[str, Any]]
    best_scenario: Dict[str, Any]
    race_position_delta: int
    time_delta: float
    strategy_advice: StrategyAdviceResponse

# API endpoints
@app.get("/")
async def root():
    return {"message": "F1 Strategy Predictor API is running"}

@app.post("/predict/undercut", response_model=PredictionResponse)
async def predict_undercut(request: UndercutRequest):
    try:
        result = strategy_predictor.predict_undercut(
            tire_delta=request.tire_delta,
            pace_dropoff=request.pace_dropoff,
            track_gap=request.track_gap,
            tire_deg_curve=request.tire_deg_curve,
            rival_pit_window=request.rival_pit_window
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/overcut", response_model=PredictionResponse)
async def predict_overcut(request: OvercutRequest):
    try:
        result = strategy_predictor.predict_overcut(
            tire_delta=request.tire_delta,
            pace_dropoff=request.pace_dropoff,
            track_gap=request.track_gap,
            tire_deg_curve=request.tire_deg_curve,
            rival_pit_window=request.rival_pit_window
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_scenarios(request: SimulationRequest):
    try:
        # Get race simulation results
        simulation_result = race_simulator.simulate_scenarios(
            current_lap=request.current_lap,
            total_laps=request.total_laps,
            current_position=request.current_position,
            gap_ahead=request.gap_ahead,
            gap_behind=request.gap_behind,
            current_tire_age=request.current_tire_age,
            current_compound=request.current_compound,
            weather_condition=request.weather_condition,
            pit_scenarios=request.pit_scenarios
        )
        
        # Get strategy advice from Groq's API
        strategy_advice = strategy_advisor.get_strategy_advice({
            "current_lap": request.current_lap,
            "total_laps": request.total_laps,
            "current_position": request.current_position,
            "gap_ahead": request.gap_ahead,
            "gap_behind": request.gap_behind,
            "current_tire_age": request.current_tire_age,
            "current_compound": request.current_compound,
            "weather_condition": request.weather_condition,
            "pit_scenarios": request.pit_scenarios
        })
        
        # Combine results
        result = {
            "scenarios": simulation_result["scenarios"],
            "best_scenario": simulation_result["best_scenario"],
            "race_position_delta": simulation_result["race_position_delta"],
            "time_delta": simulation_result["time_delta"],
            "strategy_advice": strategy_advice
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with uvicorn when this file is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)