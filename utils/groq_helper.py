class StrategyAssistant:
    def _format_strategy(self, strategy: Dict[str, Any]) -> str:
        """Format a race strategy into a readable string"""
        return f"""
        Pit Stops: {strategy.get('pit_stops', 'N/A')}
        Tire Compounds: {strategy.get('tire_compounds', 'N/A')}
        Stop Laps: {strategy.get('stop_laps', 'N/A')}
        Expected Pace: {strategy.get('expected_pace', 'N/A')} seconds per lap
        Weather Strategy: {strategy.get('weather_strategy', 'N/A')}
        Risk Level: {strategy.get('risk_level', 'N/A')}
        """
    def generate_race_debrief(self, strategies: List[Dict[str, Any]]) -> str:
        """Generate a race debrief summarizing strategies and missed opportunities"""
        debrief = "Race Debrief:\n"
        for strategy in strategies:
            debrief += f"Strategy: {self._format_strategy(strategy)}\n"
            # Add analysis of missed opportunities or potential improvements
            debrief += "Missed Opportunities: Analyze potential improvements here.\n"
        return debrief