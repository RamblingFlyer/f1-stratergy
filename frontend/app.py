import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_helper import StrategyAssistant

# Constants
API_URL = "http://localhost:8000"
TIRE_COMPOUNDS = ["soft", "medium", "hard", "intermediate", "wet"]
WEATHER_CONDITIONS = ["dry", "mixed", "wet"]

# Initialize the OpenAI strategy assistant
strategy_assistant = StrategyAssistant()

# Page configuration
st.set_page_config(
    page_title="F1 Strategy Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e10600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #15151e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-probability {
        font-size: 2rem;
        font-weight: bold;
    }
    .high-probability {
        color: #00a651;
    }
    .medium-probability {
        color: #ffd100;
    }
    .low-probability {
        color: #e10600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">F1 Strategy Predictor</h1>', unsafe_allow_html=True)
st.markdown("A machine learning and generative AI-powered system for predicting Formula 1 pit stop strategies")

# Sidebar for AI Strategy Assistant
st.sidebar.markdown("## AI Strategy Assistant")
st.sidebar.markdown("Ask questions about F1 strategy or get advice based on the current race situation.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.sidebar.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.sidebar.chat_input("Ask about F1 strategy..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.sidebar.chat_message("user"):
        st.markdown(prompt)
    
    # Get race context if available
    race_context = None
    if "race_situation" in st.session_state:
        race_context = st.session_state.race_situation
    
    # Get AI response
    with st.sidebar.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = strategy_assistant.answer_strategy_question(prompt, race_context)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Main content
tabs = st.tabs(["Strategy Predictor", "Scenario Simulator", "What-If Analysis", "Race Debrief"])

# Strategy Predictor Tab
with tabs[0]:
    st.markdown('<h2 class="sub-header">Undercut/Overcut Predictor</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Race Parameters")
        
        # Input parameters
        tire_delta = st.slider("Tire Age Delta (laps)", 0, 20, 5, 
                             help="Difference in tire age between your car and rival")
        pace_dropoff = st.slider("Pace Drop-off (seconds/lap)", 0.0, 1.0, 0.2, 0.1,
                               help="How much lap time is lost per lap due to tire degradation")
        track_gap = st.slider("Track Gap (seconds)", 0.0, 5.0, 1.5, 0.1,
                            help="Gap to the car ahead/behind in seconds")
        tire_deg_curve = st.slider("Tire Degradation Curve", 0.0, 3.0, 1.0, 0.1,
                                 help="How aggressive is the tire degradation (higher = more aggressive)")
        rival_pit_window = st.slider("Rival Pit Window (laps)", 0, 10, 3,
                                   help="Expected number of laps until rival pits")
        
        # Store race situation for AI assistant context
        st.session_state.race_situation = {
            "tire_delta": tire_delta,
            "pace_dropoff": pace_dropoff,
            "track_gap": track_gap,
            "tire_deg_curve": tire_deg_curve,
            "rival_pit_window": rival_pit_window,
            "position": 3,  # Default values
            "current_lap": 20,
            "total_laps": 50,
            "gap_ahead": track_gap,
            "gap_behind": 2.0,
            "tire_compound": "medium",
            "tire_age": 15,
            "weather": "dry"
        }
        
        # Prediction buttons
        col1a, col1b = st.columns(2)
        with col1a:
            predict_undercut = st.button("Predict Undercut", type="primary")
        with col1b:
            predict_overcut = st.button("Predict Overcut", type="primary")
    
    with col2:
        st.markdown("### Prediction Results")
        
        # Make prediction when button is clicked
        if predict_undercut or predict_overcut:
            with st.spinner("Calculating prediction..."):
                try:
                    # Prepare request data
                    data = {
                        "tire_delta": tire_delta,
                        "pace_dropoff": pace_dropoff,
                        "track_gap": track_gap,
                        "tire_deg_curve": tire_deg_curve,
                        "rival_pit_window": rival_pit_window
                    }
                    
                    # Make API request
                    if predict_undercut:
                        response = requests.post(f"{API_URL}/predict/undercut", json=data)
                        strategy_type = "Undercut"
                    else:  # predict_overcut
                        response = requests.post(f"{API_URL}/predict/overcut", json=data)
                        strategy_type = "Overcut"
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.markdown(f"#### {strategy_type} Strategy Analysis")
                        
                        # Success probability with color coding
                        prob = result["success_probability"]
                        prob_class = "high-probability" if prob > 0.7 else "medium-probability" if prob > 0.4 else "low-probability"
                        st.markdown(f"<div class='prediction-box'>"
                                  f"<p>Success Probability:</p>"
                                  f"<p class='success-probability {prob_class}'>{prob:.1%}</p>"
                                  f"<p>Confidence: {result['confidence_score']:.1%}</p>"
                                  f"<p><strong>Recommendation:</strong> {result['recommended_action']}</p>"
                                  f"</div>", unsafe_allow_html=True)
                        
                        # Factor importance visualization
                        st.markdown("#### Factor Importance")
                        factors_df = pd.DataFrame({
                            'Factor': list(result['factors'].keys()),
                            'Importance': list(result['factors'].values())
                        })
                        
                        fig = px.bar(factors_df, x='Importance', y='Factor', orientation='h',
                                   color='Importance', color_continuous_scale='Viridis')
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Get AI analysis
                        with st.spinner("Getting AI analysis..."):
                            ai_analysis = strategy_assistant.get_strategy_advice(st.session_state.race_situation)
                            st.markdown("#### AI Strategy Analysis")
                            st.markdown(ai_analysis)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

# Scenario Simulator Tab
with tabs[1]:
    st.markdown('<h2 class="sub-header">Race Scenario Simulator</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Race Parameters")
        
        # Race situation inputs
        current_lap = st.number_input("Current Lap", min_value=1, max_value=100, value=20)
        total_laps = st.number_input("Total Laps", min_value=current_lap, max_value=100, value=50)
        current_position = st.number_input("Current Position", min_value=1, max_value=20, value=3)
        gap_ahead = st.number_input("Gap to Car Ahead (seconds)", min_value=0.0, max_value=30.0, value=1.5, step=0.1)
        gap_behind = st.number_input("Gap to Car Behind (seconds)", min_value=0.0, max_value=30.0, value=2.0, step=0.1)
        current_tire_age = st.number_input("Current Tire Age (laps)", min_value=0, max_value=50, value=15)
        current_compound = st.selectbox("Current Tire Compound", TIRE_COMPOUNDS, index=1)  # Default to medium
        weather_condition = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=0)  # Default to dry
        
        # Pit scenarios
        st.markdown("### Pit Scenarios to Simulate")
        
        # Default scenarios
        if "pit_scenarios" not in st.session_state:
            st.session_state.pit_scenarios = [
                {"name": "Stay out", "pit_lap": total_laps + 1, "new_compound": ""},
                {"name": "Pit now", "pit_lap": current_lap, "new_compound": "hard"},
                {"name": "Pit in 3 laps", "pit_lap": current_lap + 3, "new_compound": "hard"},
                {"name": "Pit in 5 laps", "pit_lap": current_lap + 5, "new_compound": "medium"}
            ]
        
        # Display and edit scenarios
        for i, scenario in enumerate(st.session_state.pit_scenarios):
            st.markdown(f"**Scenario {i+1}: {scenario['name']}**")
            col_a, col_b = st.columns(2)
            with col_a:
                scenario['pit_lap'] = st.number_input(f"Pit Lap", 
                                                   min_value=current_lap, 
                                                   max_value=total_laps + 1,  # +1 for "no pit"
                                                   value=scenario['pit_lap'],
                                                   key=f"pit_lap_{i}")
            with col_b:
                if scenario['pit_lap'] <= total_laps:  # Only show compound if actually pitting
                    scenario['new_compound'] = st.selectbox(f"New Compound", 
                                                         TIRE_COMPOUNDS,
                                                         index=TIRE_COMPOUNDS.index(scenario['new_compound']) if scenario['new_compound'] in TIRE_COMPOUNDS else 2,
                                                         key=f"compound_{i}")
        
        # Add new scenario button
        if st.button("Add Scenario"):
            st.session_state.pit_scenarios.append({
                "name": f"Scenario {len(st.session_state.pit_scenarios) + 1}",
                "pit_lap": current_lap + 2,
                "new_compound": "medium"
            })
            st.experimental_rerun()
        
        # Run simulation button
        simulate_button = st.button("Run Simulation", type="primary")
    
    with col2:
        if simulate_button:
            with st.spinner("Running simulation..."):
                try:
                    # Prepare request data
                    data = {
                        "current_lap": current_lap,
                        "total_laps": total_laps,
                        "current_position": current_position,
                        "gap_ahead": gap_ahead,
                        "gap_behind": gap_behind,
                        "current_tire_age": current_tire_age,
                        "current_compound": current_compound,
                        "weather_condition": weather_condition,
                        "pit_scenarios": st.session_state.pit_scenarios
                    }
                    
                    # Update race situation for AI assistant
                    st.session_state.race_situation.update({
                        "current_lap": current_lap,
                        "total_laps": total_laps,
                        "position": current_position,
                        "gap_ahead": gap_ahead,
                        "gap_behind": gap_behind,
                        "tire_compound": current_compound,
                        "tire_age": current_tire_age,
                        "weather": weather_condition
                    })
                    
                    # Make API request
                    response = requests.post(f"{API_URL}/simulate", json=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.markdown("### Simulation Results")
                        
                        # Best scenario highlight
                        best = result["best_scenario"]
                        st.markdown(f"**Best Strategy: {best['name']}**")
                        st.markdown(f"Expected position change: {'+' if best['position_delta'] > 0 else ''}{best['position_delta']}")
                        st.markdown(f"Final position: P{best['final_position']}")
                        
                        # Create comparison table
                        scenarios_df = pd.DataFrame(result["scenarios"])
                        
                        # Format the table
                        display_cols = ['name', 'pit_lap', 'new_compound', 'total_race_time', 'position_delta', 'final_position']
                        if 'safety_car_appeared' in scenarios_df.columns:
                            display_cols.append('safety_car_appeared')
                        
                        formatted_df = scenarios_df[display_cols].copy()
                        formatted_df.columns = ['Strategy', 'Pit Lap', 'Compound', 'Race Time', 'Pos. Delta', 'Final Pos.', 'SC']
                        
                        # Round numeric values
                        if 'Race Time' in formatted_df.columns:
                            formatted_df['Race Time'] = formatted_df['Race Time'].round(2)
                        
                        # Add visual indicators
                        formatted_df['Pos. Delta'] = formatted_df['Pos. Delta'].apply(
                            lambda x: f"+{x}" if x > 0 else str(x))
                        
                        st.dataframe(formatted_df, use_container_width=True)
                        
                        # Lap time visualization
                        st.markdown("### Lap Time Comparison")
                        
                        # Create lap time chart
                        fig = go.Figure()
                        
                        for scenario in result["scenarios"]:
                            if 'lap_times' in scenario:
                                fig.add_trace(go.Scatter(
                                    x=list(range(current_lap, current_lap + len(scenario['lap_times']))),
                                    y=scenario['lap_times'],
                                    mode='lines',
                                    name=scenario['name']
                                ))
                        
                        fig.update_layout(
                            title="Predicted Lap Times by Strategy",
                            xaxis_title="Lap Number",
                            yaxis_title="Lap Time (seconds)",
                            legend_title="Strategy",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Get AI analysis
                        with st.spinner("Getting AI analysis..."):
                            ai_analysis = strategy_assistant.get_strategy_advice(st.session_state.race_situation)
                            st.markdown("### AI Strategy Analysis")
                            st.markdown(ai_analysis)
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")

# What-If Analysis Tab
with tabs[2]:
    st.markdown('<h2 class="sub-header">Alternate Timeline Generator</h2>', unsafe_allow_html=True)
    st.markdown("Compare actual race strategy with alternative scenarios to see what might have happened.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Actual Strategy")
        
        # Actual strategy inputs
        actual_strategy = {}
        actual_strategy["race_name"] = st.text_input("Race Name", "Monaco Grand Prix")
        actual_strategy["year"] = st.number_input("Year", min_value=2000, max_value=2023, value=2023)
        actual_strategy["pit_timing"] = st.text_input("Pit Stop Timing", "Lap 22, Lap 45")
        actual_strategy["tire_compounds"] = st.text_input("Tire Compounds Used", "Soft → Medium → Hard")
        actual_strategy["start_position"] = st.number_input("Starting Position", min_value=1, max_value=20, value=3)
        actual_strategy["finish_position"] = st.number_input("Finishing Position", min_value=1, max_value=20, value=2)
        actual_strategy["key_moments"] = st.text_area("Key Race Moments", "Safety car on lap 30, rain started on lap 40")
    
    with col2:
        st.markdown("### Alternative Strategy")
        
        # Alternative strategy inputs
        alternate_strategy = {}
        alternate_strategy["pit_timing"] = st.text_input("Alternative Pit Stop Timing", "Lap 18, Lap 42")
        alternate_strategy["tire_compounds"] = st.text_input("Alternative Tire Compounds", "Medium → Hard → Soft")
        alternate_strategy["start_position"] = actual_strategy["start_position"]  # Same starting position
        alternate_strategy["finish_position"] = st.number_input("Estimated Finishing Position", min_value=1, max_value=20, value=1)
        alternate_strategy["key_moments"] = st.text_area("Key Strategic Differences", "Earlier first stop to undercut, softs for final stint")
    
    # Generate button
    if st.button("Generate Alternate Timeline", type="primary"):
        with st.spinner("Generating alternate timeline..."):
            try:
                # Call OpenAI to generate the alternate timeline
                alternate_timeline = strategy_assistant.generate_alternate_timeline(
                    actual_strategy, alternate_strategy)
                
                # Display the result
                st.markdown("### What-If Scenario Analysis")
                st.markdown(alternate_timeline)
            except Exception as e:
                st.error(f"Error generating alternate timeline: {str(e)}")

# Race Debrief Tab
with tabs[3]:
    st.markdown('<h2 class="sub-header">Post-Race Debrief Generator</h2>', unsafe_allow_html=True)
    st.markdown("Generate a comprehensive post-race analysis of strategy decisions and outcomes.")
    
    # Race data inputs
    race_data = {}
    race_data["race_name"] = st.text_input("Race", "British Grand Prix")
    race_data["year"] = st.number_input("Season", min_value=2000, max_value=2023, value=2023)
    race_data["track"] = st.text_input("Track", "Silverstone")
    race_data["weather"] = st.text_input("Weather Conditions", "Started dry, rain from lap 40-52")
    
    col1, col2 = st.columns(2)
    
    with col1:
        race_data["start_position"] = st.number_input("Driver Starting Position", min_value=1, max_value=20, value=4)
        race_data["finish_position"] = st.number_input("Driver Finishing Position", min_value=1, max_value=20, value=1)
    
    with col2:
        race_data["pit_stops"] = st.text_input("Pit Stops", "Lap 18 (Medium), Lap 42 (Intermediate)")
        race_data["tire_strategy"] = st.text_input("Tire Strategy", "Soft → Medium → Intermediate")
    
    race_data["race_summary"] = st.text_area("Race Summary", 
                                        "Started P4, maintained position in first stint. Undercut worked on lap 18 gaining P3. " +
                                        "Rain started on lap 40, pitted for intermediates on lap 42 while others stayed out. " +
                                        "Gained P1 as others struggled on slicks in wet conditions and maintained lead until finish.")
    
    # Generate button
    if st.button("Generate Race Debrief", type="primary"):
        with st.spinner("Generating race debrief..."):
            try:
                # Call OpenAI to generate the debrief
                debrief = strategy_assistant.generate_race_debrief(race_data)
                
                # Display the result
                st.markdown("### Post-Race Strategy Debrief")
                st.markdown(debrief)
            except Exception as e:
                st.error(f"Error generating race debrief: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>F1 Strategy Predictor | Powered by Machine Learning and GPT-4</p>
</div>
""", unsafe_allow_html=True)