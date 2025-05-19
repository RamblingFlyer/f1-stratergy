import os
from typing import Dict, Any, List
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StrategyAdvisor:
    """Class to handle strategy advice using Groq's API"""
    
    def __init__(self):
        # Initialize Groq API client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.client = Groq(api_key=api_key)
        self.model = "valid-model-name"
        
    def get_strategy_advice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get strategy advice from Groq's API
        
        Args:
            context: Dictionary containing race situation context
                - current_lap: int
                - total_laps: int
                - current_position: int
                - gap_ahead: float
                - gap_behind: float
                - current_tire_age: int
                - current_compound: str
                - weather_condition: str
                - pit_scenarios: List[Dict]
        
        Returns:
            Dictionary containing strategy advice
        """
        try:
            # Format the prompt for Groq's API
            prompt = f"""
            You are an F1 strategy advisor. Analyze the following race situation and provide strategy advice:
            
            Current Lap: {context['current_lap']} of {context['total_laps']}
            Position: {context['current_position']}
            Gap to car ahead: {context['gap_ahead']} seconds
            Gap to car behind: {context['gap_behind']} seconds
            Current tire age: {context['current_tire_age']} laps
            Current compound: {context['current_compound']}
            Weather: {context['weather_condition']}
            
            Consider these pit scenarios:
            {self._format_pit_scenarios(context['pit_scenarios'])}
            
            Please provide:
            1. Recommended pit strategy
            2. Probability of success
            3. Key factors influencing the decision
            """
            
            # Create chat completion
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert F1 strategy advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Get the response
            response = completion.choices[0].message.content
            
            # Parse the response
            advice = self._parse_advice(response)
            
            return advice
            
        except Exception as e:
            raise Exception(f"Error getting strategy advice: {str(e)}")
    
    def _format_pit_scenarios(self, scenarios: List[Dict]) -> str:
        """Format pit scenarios for the prompt"""
        formatted = []
        for i, scenario in enumerate(scenarios, 1):
            formatted.append(f"Scenario {i}:")
            formatted.append(f"- Lap: {scenario.get('lap', 'N/A')}")
            formatted.append(f"- Compound: {scenario.get('compound', 'N/A')}")
            formatted.append(f"- Expected duration: {scenario.get('duration', 'N/A')} laps")
        return "\n".join(formatted)
    
    def _parse_advice(self, response: str) -> Dict:
        """Parse the API response into a structured format"""
        # Initialize the advice dictionary with default values
        advice = {
            "recommended_strategy": "",
            "success_probability": 0.0,
            "key_factors": {}
        }
        
        try:
            # Split the response into lines and process each line
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                # Extract recommended strategy
                if line.startswith("1.") or "Recommended" in line:
                    advice["recommended_strategy"] = line.split(".", 1)[-1].strip()
                
                # Extract probability of success
                elif line.startswith("2.") or "Probability" in line:
                    # Look for percentage in the line
                    prob_text = line.split(":")[-1].strip() if ":" in line else line
                    # Find the first number in the text
                    import re
                    prob_match = re.search(r'\d+', prob_text)
                    if prob_match:
                        prob_value = float(prob_match.group())
                        advice["success_probability"] = prob_value / 100
                
                # Extract key factors
                elif line.startswith("3.") or "Key factors" in line.lower() or "Factors" in line:
                    # Try to extract factors and their weights
                    try:
                        factors_text = line.split(":", 1)[-1].strip()
                        # Split factors by comma or bullet points
                        factor_items = [f.strip() for f in re.split(r'[,•-]', factors_text) if f.strip()]
                        
                        for factor in factor_items:
                            # Try to extract weight if present in parentheses
                            weight_match = re.search(r'\(?(\d+)%?\)?', factor)
                            if weight_match:
                                weight = float(weight_match.group(1)) / 100
                                name = factor[:weight_match.start()].strip()
                            else:
                                # If no weight is found, assign equal weights
                                weight = 1.0 / len(factor_items)
                                name = factor.strip()
                            
                            advice["key_factors"][name] = weight
                    except Exception:
                        # If factor parsing fails, continue with other lines
                        continue
            
            # Normalize key factor weights if any exist
            if advice["key_factors"]:
                total_weight = sum(advice["key_factors"].values())
                if total_weight > 0:
                    advice["key_factors"] = {k: v/total_weight for k, v in advice["key_factors"].items()}
            
        except Exception as e:
            print(f"Error parsing advice: {str(e)}")
        
        return advice
