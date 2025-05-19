import numpy as np
import random
from typing import Dict, Any, List, Tuple

class RaceSimulator:
    """Simulation engine for F1 race strategies"""
    
    def __init__(self):
        # Tire compound performance characteristics (base pace in seconds)
        self.tire_compounds = {
            "soft": {"base_pace": 0.0, "deg_rate": 0.05, "optimal_window": (1, 15)},
            "medium": {"base_pace": 0.5, "deg_rate": 0.03, "optimal_window": (10, 35)},
            "hard": {"base_pace": 1.0, "deg_rate": 0.02, "optimal_window": (25, 50)},
            "intermediate": {"base_pace": 3.0, "deg_rate": 0.04, "optimal_window": (1, 30)},
            "wet": {"base_pace": 6.0, "deg_rate": 0.01, "optimal_window": (1, 40)}
        }
        
        # Pit stop time loss in seconds (includes entry, stop, and exit)
        self.pit_stop_time = 22.0
        
        # Track position worth in seconds
        self.position_worth = 1.5  # Each position is worth about 1.5 seconds
        
    def _calculate_lap_time(self, compound: str, tire_age: int, weather: str) -> float:
        """Calculate expected lap time based on tire compound, age, and weather"""
        base_time = 90.0  # Base lap time in seconds
        
        # Add compound-specific pace
        compound_data = self.tire_compounds[compound]
        time = base_time + compound_data["base_pace"]
        
        # Add degradation based on tire age
        # Simplified model: linear degradation with compound-specific rate
        time += tire_age * compound_data["deg_rate"]
        
        # Weather effects
        if weather == "wet" and compound not in ["intermediate", "wet"]:
            time += 5.0  # Major penalty for dry tires in wet conditions
        elif weather == "mixed":
            if compound in ["intermediate"]:
                time += 1.0  # Slight penalty for inters in mixed conditions
            elif compound in ["wet"]:
                time += 2.0  # Bigger penalty for full wets in mixed conditions
            else:
                time += 3.0  # Penalty for dry tires in mixed conditions
        elif weather == "dry" and compound in ["intermediate", "wet"]:
            time += 4.0  # Penalty for wet tires in dry conditions
            
        # Add some randomness (+-0.3s)
        time += random.uniform(-0.3, 0.3)
        
        return time
    
    def _simulate_stint(self, start_lap: int, end_lap: int, compound: str, 
                       start_tire_age: int, weather: str) -> Tuple[float, List[float]]:
        """Simulate a race stint and return total time and lap times"""
        total_time = 0.0
        lap_times = []
        
        current_tire_age = start_tire_age
        
        for lap in range(start_lap, end_lap + 1):
            lap_time = self._calculate_lap_time(compound, current_tire_age, weather)
            total_time += lap_time
            lap_times.append(lap_time)
            current_tire_age += 1
            
        return total_time, lap_times
    
    def _evaluate_scenario(self, scenario: Dict[str, Any], race_params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single pit stop scenario and return results"""
        # Extract race parameters
        current_lap = race_params["current_lap"]
        total_laps = race_params["total_laps"]
        current_position = race_params["current_position"]
        gap_ahead = race_params["gap_ahead"]
        gap_behind = race_params["gap_behind"]
        current_tire_age = race_params["current_tire_age"]
        current_compound = race_params["current_compound"]
        weather_condition = race_params["weather_condition"]
        
        # Extract scenario parameters
        pit_lap = scenario.get("pit_lap", current_lap)
        new_compound = scenario.get("new_compound", "medium")
        expected_weather_change = scenario.get("weather_change", None)
        safety_car_probability = scenario.get("safety_car_probability", 0.0)
        
        # Initialize result
        result = scenario.copy()
        result["total_race_time"] = 0.0
        result["position_delta"] = 0
        result["lap_times"] = []
        
        # Simulate first stint (current tires until pit lap)
        if pit_lap > current_lap:
            first_stint_time, first_stint_lap_times = self._simulate_stint(
                current_lap, pit_lap - 1, 
                current_compound, current_tire_age, 
                weather_condition
            )
            result["total_race_time"] += first_stint_time
            result["lap_times"].extend(first_stint_lap_times)
        
        # Add pit stop time if pitting
        if pit_lap <= total_laps:
            result["total_race_time"] += self.pit_stop_time
            
            # Simulate second stint (new tires from pit lap to end)
            # Check for weather changes
            second_stint_weather = weather_condition
            if expected_weather_change and expected_weather_change["lap"] >= pit_lap:
                second_stint_weather = expected_weather_change["condition"]
                
            second_stint_time, second_stint_lap_times = self._simulate_stint(
                pit_lap, total_laps,
                new_compound, 0,  # New tires, so age is 0
                second_stint_weather
            )
            result["total_race_time"] += second_stint_time
            result["lap_times"].extend(second_stint_lap_times)
        
        # Simulate safety car if applicable
        if safety_car_probability > 0 and random.random() < safety_car_probability:
            # Safety car typically bunches up the field, reducing gaps
            sc_lap = random.randint(current_lap + 1, total_laps)
            result["safety_car_appeared"] = True
            result["safety_car_lap"] = sc_lap
            
            # If safety car appears before planned pit, adjust strategy
            if sc_lap < pit_lap:
                # Recalculate with earlier pit stop during SC
                adjusted_result = self._evaluate_scenario(
                    {**scenario, "pit_lap": sc_lap},
                    race_params
                )
                # Safety car pit stop is faster (less time lost)
                adjusted_result["total_race_time"] -= 5.0  # Save ~5s under SC
                return adjusted_result
        else:
            result["safety_car_appeared"] = False
        
        # Estimate position changes based on time gaps
        # This is a simplified model and would be more complex in reality
        time_delta_ahead = result["total_race_time"] - gap_ahead
        time_delta_behind = gap_behind - result["total_race_time"]
        
        # Positive means we're faster (gaining), negative means we're slower (losing)
        if time_delta_ahead > 0:
            result["position_delta"] += 1  # Overtake car ahead
        if time_delta_behind < 0:
            result["position_delta"] -= 1  # Lose position to car behind
            
        # Calculate final position
        result["final_position"] = current_position - result["position_delta"]
        
        return result
    
    def simulate_scenarios(self, **kwargs) -> Dict[str, Any]:
        """Simulate multiple pit stop scenarios and return the best one"""
        # Extract race parameters
        race_params = {
            "current_lap": kwargs["current_lap"],
            "total_laps": kwargs["total_laps"],
            "current_position": kwargs["current_position"],
            "gap_ahead": kwargs["gap_ahead"],
            "gap_behind": kwargs["gap_behind"],
            "current_tire_age": kwargs["current_tire_age"],
            "current_compound": kwargs["current_compound"],
            "weather_condition": kwargs["weather_condition"]
        }
        
        # Get scenarios to simulate
        pit_scenarios = kwargs["pit_scenarios"]
        
        # If no scenarios provided, create some default ones
        if not pit_scenarios:
            pit_scenarios = [
                {"name": "Stay out", "pit_lap": kwargs["total_laps"] + 1},  # No pit stop
                {"name": "Pit now", "pit_lap": kwargs["current_lap"], "new_compound": "hard"},
                {"name": "Pit in 3 laps", "pit_lap": kwargs["current_lap"] + 3, "new_compound": "hard"},
                {"name": "Pit in 5 laps", "pit_lap": kwargs["current_lap"] + 5, "new_compound": "hard"}
            ]
        
        # Simulate each scenario
        results = []
        for scenario in pit_scenarios:
            result = self._evaluate_scenario(scenario, race_params)
            results.append(result)
        
        # Find the best scenario (lowest race time)
        best_scenario = min(results, key=lambda x: x["total_race_time"])
        
        # Calculate time delta between best and worst scenario
        worst_scenario = max(results, key=lambda x: x["total_race_time"])
        time_delta = worst_scenario["total_race_time"] - best_scenario["total_race_time"]
        
        return {
            "scenarios": results,
            "best_scenario": best_scenario,
            "race_position_delta": best_scenario["position_delta"],
            "time_delta": time_delta
        }