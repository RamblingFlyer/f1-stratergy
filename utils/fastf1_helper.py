import fastf1
import pandas as pd
from typing import Dict, Any, List, Optional

class F1DataHelper:
    """Helper class for fetching and processing F1 data using FastF1 API"""
    
    def __init__(self):
        """Initialize the F1 data helper"""
        # Enable caching for faster subsequent data loading
        fastf1.Cache.enable_cache('cache')
    
    def load_session(self, year: int, race: str, session_type: str = 'R') -> fastf1.core.Session:
        """Load a specific F1 session
        
        Args:
            year: Season year (e.g., 2023)
            race: Race name or round number
            session_type: Session type ('R' for race, 'Q' for qualifying, 'FP1', 'FP2', 'FP3' for practice)
            
        Returns:
            Session object
        """
        try:
            session = fastf1.get_session(year, race, session_type)
            session.load()
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None
    
    def get_driver_lap_times(self, session: fastf1.core.Session, driver: str) -> pd.DataFrame:
        """Get lap times for a specific driver
        
        Args:
            session: Loaded session object
            driver: Driver identifier (e.g., 'VER', 'HAM')
            
        Returns:
            DataFrame with lap times
        """
        try:
            laps = session.laps.pick_driver(driver)
            return laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife', 'FreshTyre', 'Team']]
        except Exception as e:
            print(f"Error getting lap times: {e}")
            return pd.DataFrame()
    
    def get_tire_strategies(self, session: fastf1.core.Session) -> Dict[str, List[Dict[str, Any]]]:
        """Get tire strategies for all drivers in a race
        
        Args:
            session: Loaded session object
            
        Returns:
            Dictionary mapping driver codes to their tire strategies
        """
        try:
            # Get all drivers
            drivers = session.drivers
            strategies = {}
            
            for driver in drivers:
                driver_laps = session.laps.pick_driver(driver)
                driver_stints = []
                
                # Group by stint
                for stint, stint_data in driver_laps.groupby('Stint'):
                    # Get first lap of stint to determine compound
                    first_lap = stint_data.iloc[0]
                    stint_info = {
                        'stint': int(stint),
                        'compound': first_lap['Compound'],
                        'start_lap': int(first_lap['LapNumber']),
                        'end_lap': int(stint_data.iloc[-1]['LapNumber']),
                        'laps': len(stint_data)
                    }
                    driver_stints.append(stint_info)
                
                strategies[driver] = driver_stints
            
            return strategies
        except Exception as e:
            print(f"Error getting tire strategies: {e}")
            return {}
    
    def get_race_pace_comparison(self, session: fastf1.core.Session, 
                               drivers: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare race pace between drivers
        
        Args:
            session: Loaded session object
            drivers: List of driver identifiers to compare
            
        Returns:
            Dictionary with pace comparison data
        """
        try:
            pace_data = {}
            
            for driver in drivers:
                driver_laps = session.laps.pick_driver(driver)
                
                # Calculate median lap time by compound (excluding outliers)
                pace_by_compound = {}
                for compound, compound_laps in driver_laps.groupby('Compound'):
                    # Convert lap times to seconds and filter out outliers
                    lap_times = compound_laps['LapTime'].dt.total_seconds()
                    valid_times = lap_times[(lap_times > lap_times.quantile(0.05)) & 
                                           (lap_times < lap_times.quantile(0.95))]
                    
                    if not valid_times.empty:
                        pace_by_compound[compound] = valid_times.median()
                
                pace_data[driver] = pace_by_compound
            
            return pace_data
        except Exception as e:
            print(f"Error calculating race pace: {e}")
            return {}
    
    def get_undercut_potential(self, session: fastf1.core.Session, 
                             lap_number: int, driver: str, 
                             target_driver: str) -> Dict[str, Any]:
        """Calculate undercut potential between two drivers at a specific lap
        
        Args:
            session: Loaded session object
            lap_number: Current lap number
            driver: Driver considering the undercut
            target_driver: Driver ahead (target of undercut)
            
        Returns:
            Dictionary with undercut analysis
        """
        try:
            # Get relevant lap data
            driver_laps = session.laps.pick_driver(driver)
            target_laps = session.laps.pick_driver(target_driver)
            
            # Get current lap for both drivers
            current_driver_lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
            current_target_lap = target_laps[target_laps['LapNumber'] == lap_number].iloc[0]
            
            # Get gap between drivers
            gap = current_target_lap['LapTime'].total_seconds() - current_driver_lap['LapTime'].total_seconds()
            
            # Get current tire compounds and age
            driver_compound = current_driver_lap['Compound']
            target_compound = current_target_lap['Compound']
            driver_tire_age = current_driver_lap['TyreLife']
            target_tire_age = current_target_lap['TyreLife']
            
            # Estimate new tire advantage (simplified)
            new_tire_advantage = 0.5 * driver_tire_age  # Rough estimate: 0.5s per lap of tire age
            
            # Estimate pit stop time loss
            pit_time_loss = 22.0  # Standard pit stop time loss
            
            # Calculate undercut potential
            undercut_potential = new_tire_advantage - pit_time_loss + gap
            
            return {
                'gap': gap,
                'driver_compound': driver_compound,
                'target_compound': target_compound,
                'driver_tire_age': driver_tire_age,
                'target_tire_age': target_tire_age,
                'new_tire_advantage': new_tire_advantage,
                'pit_time_loss': pit_time_loss,
                'undercut_potential': undercut_potential,
                'recommendation': 'Undercut possible' if undercut_potential > 0 else 'Undercut not recommended'
            }
        except Exception as e:
            print(f"Error calculating undercut potential: {e}")
            return {}
    
    def get_track_position_data(self, session: fastf1.core.Session, 
                              lap_number: int) -> pd.DataFrame:
        """Get track position data for all drivers at a specific lap
        
        Args:
            session: Loaded session object
            lap_number: Lap number to analyze
            
        Returns:
            DataFrame with track position data
        """
        try:
            # Get all laps for the specified lap number
            lap_data = session.laps[session.laps['LapNumber'] == lap_number]
            
            # Sort by position
            lap_data = lap_data.sort_values('Position')
            
            # Select relevant columns
            position_data = lap_data[['Driver', 'Position', 'LapTime', 'Compound', 'TyreLife', 'Team']]
            
            # Calculate gaps between consecutive positions
            position_data['GapToNext'] = position_data['LapTime'].diff(-1).dt.total_seconds() * -1
            
            return position_data
        except Exception as e:
            print(f"Error getting track position data: {e}")
            return pd.DataFrame()