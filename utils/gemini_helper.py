import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from groq import GroqApi

# Load environment variables
load_dotenv()

# Initialize Groq API client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = GroqApi(api_key)

class StrategyAssistant:
    """LLM-powered F1 strategy assistant using Groq"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """Initialize the strategy assistant with specified model"""
        self.model = model
        self.system_prompt = """
        You are an expert Formula 1 race engineer and strategist. 
        Your job is to analyze race situations and provide strategic advice to the driver and team.
        You have deep knowledge of F1 regulations, tire compounds, race tactics, and historical race data.
        Provide clear, concise, and data-backed recommendations.
        """
        
    def answer_strategy_question(self, question: str, race_context: Optional[Dict[str, Any]] = None) -> str:
        """Answer a strategy-related question with optional race context"""
        try:
            # Format the prompt with context if available
            prompt = f"""
            You are an expert F1 strategist. Answer the following question:
            {question}
            
            Race Context:
            {self._format_race_context(race_context) if race_context else "No specific race context provided"}
            """
            
            # Initialize Groq model
            chat = groq.Chat(model=self.model, system_prompt=self.system_prompt)
            
            # Generate response
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"Error answering question: {str(e)}"

    def _format_race_context(self, race_context: Dict[str, Any]) -> str:
        """Format race context into a readable string"""
        return f"""
        Current Lap: {race_context.get('current_lap', 'N/A')}
        Total Laps: {race_context.get('total_laps', 'N/A')}
        Position: {race_context.get('current_position', 'N/A')}
        Gap Ahead: {race_context.get('gap_ahead', 'N/A')} seconds
        Gap Behind: {race_context.get('gap_behind', 'N/A')} seconds
        Tire Age: {race_context.get('current_tire_age', 'N/A')} laps
        Compound: {race_context.get('current_compound', 'N/A')}
        Weather: {race_context.get('weather_condition', 'N/A')}
        """

    def get_strategy_advice(self, race_situation: Dict[str, Any]) -> str:
        """Generate strategic advice based on the current race situation"""
        # Format the race situation into a prompt
        prompt = self._format_race_situation(race_situation)
        
        try:
            # Initialize Groq model
            chat = groq.Chat(model=self.model, system_prompt=self.system_prompt)
            
            # Generate response
            response = chat.send_message(prompt)
            return response.text
        except groq.ModelNotFoundError:
            # Fallback to a default model if the specified model is not found
            self.model = "default-model"
            chat = groq.Chat(model=self.model, system_prompt=self.system_prompt)
            try:
                response = chat.send_message(prompt)
                return response.text
            except Exception as e:
                return f"Error generating strategy advice: {str(e)}"
        except Exception as e:
            return f"Error generating strategy advice: {str(e)}"

    def generate_alternate_timeline(self, actual_strategy: Dict[str, Any], 
                                    alternate_strategy: Dict[str, Any]) -> str:
        """Generate a 'what-if' scenario comparing actual strategy with an alternate one"""
        # Create a prompt comparing the two strategies
        prompt = f"""
        Compare the following two F1 race strategies and create a detailed 'what-if' scenario 
        for how the race might have unfolded with the alternate strategy:
        
        ACTUAL STRATEGY:
        {self._format_strategy(actual_strategy)}
        
        ALTERNATE STRATEGY:
        {self._format_strategy(alternate_strategy)}
        
        Provide a narrative of how the alternate strategy might have played out, including:
        - Potential position changes
        - Tire performance differences
        - Key moments where the race outcome could have changed
        - Final result comparison
        """
        
        try:
            # Initialize Groq model
            chat = groq.Chat(model=self.model, system_prompt=self.system_prompt)
            
            # Generate response
            response = chat.send_message(prompt)
            return response.text
        except groq.ModelNotFoundError:
            # Fallback to a default model if the specified model is not found
            self.model = "default-model"
            chat = groq.Chat(model=self.model, system_prompt=self.system_prompt)
            try:
                response = chat.send_message(prompt)
                return response.text
            except Exception as e:
                return f"Error generating alternate timeline: {str(e)}"
        except Exception as e:
            return f"Error generating alternate timeline: {str(e)}"