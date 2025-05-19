import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, Tuple, List

class StrategyPredictor:
    """ML model for predicting F1 pit stop strategy success"""
    
    def __init__(self, model_path: str = None):
        """Initialize the predictor with pre-trained models or create new ones"""
        # Feature importance for each factor (will be updated when models are trained)
        self.undercut_feature_importance = {
            "tire_delta": 0.35,
            "pace_dropoff": 0.25,
            "track_gap": 0.20,
            "tire_deg_curve": 0.15,
            "rival_pit_window": 0.05
        }
        
        self.overcut_feature_importance = {
            "tire_delta": 0.30,
            "pace_dropoff": 0.30,
            "track_gap": 0.15,
            "tire_deg_curve": 0.20,
            "rival_pit_window": 0.05
        }
        
        # Initialize models
        self.undercut_model = self._create_model()
        self.overcut_model = self._create_model()
        self.scaler = StandardScaler()
        
        # Load pre-trained models if available
        if model_path and os.path.exists(model_path):
            try:
                self._load_models(model_path)
            except Exception as e:
                print(f"Error loading models: {e}. Using default models.")
    
    def _create_model(self):
        """Create a new model instance"""
        # For now, we'll use a RandomForest classifier
        # In a real implementation, this would be trained on historical data
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _load_models(self, model_path: str):
        """Load pre-trained models from disk"""
        self.undercut_model = joblib.load(os.path.join(model_path, 'undercut_model.joblib'))
        self.overcut_model = joblib.load(os.path.join(model_path, 'overcut_model.joblib'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
        
        # Load feature importances
        with open(os.path.join(model_path, 'feature_importance.json'), 'r') as f:
            import json
            importances = json.load(f)
            self.undercut_feature_importance = importances['undercut']
            self.overcut_feature_importance = importances['overcut']
    
    def _preprocess_features(self, **kwargs) -> np.ndarray:
        """Preprocess input features for model prediction"""
        # Extract features in the correct order
        features = np.array([
            kwargs['tire_delta'],
            kwargs['pace_dropoff'],
            kwargs['track_gap'],
            kwargs['tire_deg_curve'],
            kwargs['rival_pit_window']
        ]).reshape(1, -1)
        
        # In a real implementation, we would scale features
        # features = self.scaler.transform(features)
        return features
    
    def _get_confidence_score(self, probability: float) -> float:
        """Calculate confidence score based on probability"""
        # Simple mapping of probability to confidence
        # In a real implementation, this would be more sophisticated
        if probability > 0.9 or probability < 0.1:
            return 0.95  # High confidence when probability is extreme
        elif probability > 0.75 or probability < 0.25:
            return 0.80  # Good confidence
        elif probability > 0.6 or probability < 0.4:
            return 0.65  # Moderate confidence
        else:
            return 0.50  # Low confidence (near 50/50)
    
    def _get_recommended_action(self, probability: float, strategy_type: str) -> str:
        """Generate a recommended action based on prediction probability"""
        if strategy_type == "undercut":
            if probability > 0.7:
                return "Pit now for undercut attempt - high chance of success"
            elif probability > 0.5:
                return "Consider undercut - moderate chance of success"
            else:
                return "Stay out - undercut unlikely to succeed"
        else:  # overcut
            if probability > 0.7:
                return "Stay out for overcut attempt - high chance of success"
            elif probability > 0.5:
                return "Consider overcut - moderate chance of success"
            else:
                return "Pit now - overcut unlikely to succeed"
    
    def predict_undercut(self, **kwargs) -> Dict[str, Any]:
        """Predict success probability of an undercut strategy"""
        # In a real implementation, this would use the trained model
        # For now, we'll use a simplified heuristic
        
        # Preprocess features
        features = self._preprocess_features(**kwargs)
        
        # Calculate a simplified probability based on heuristics
        # In a real implementation, this would be model.predict_proba(features)[0][1]
        tire_advantage = min(1.0, kwargs['tire_delta'] / 10)  # Normalize to 0-1
        gap_factor = min(1.0, kwargs['track_gap'] / 3)  # Normalize to 0-1
        
        # Simple heuristic: undercut works better with fresher tires and smaller gaps
        probability = 0.5 + (0.3 * tire_advantage) - (0.2 * gap_factor)
        probability = max(0.05, min(0.95, probability))  # Clamp between 0.05 and 0.95
        
        # Calculate confidence score
        confidence = self._get_confidence_score(probability)
        
        # Get recommended action
        recommendation = self._get_recommended_action(probability, "undercut")
        
        return {
            "success_probability": float(probability),
            "confidence_score": float(confidence),
            "recommended_action": recommendation,
            "factors": self.undercut_feature_importance
        }
    
    def predict_overcut(self, **kwargs) -> Dict[str, Any]:
        """Predict success probability of an overcut strategy"""
        # Similar to undercut but with different heuristics
        
        # Preprocess features
        features = self._preprocess_features(**kwargs)
        
        # Calculate a simplified probability based on heuristics
        # In a real implementation, this would be model.predict_proba(features)[0][1]
        tire_advantage = min(1.0, kwargs['tire_delta'] / 10)  # Normalize to 0-1
        gap_factor = min(1.0, kwargs['track_gap'] / 3)  # Normalize to 0-1
        deg_factor = min(1.0, kwargs['tire_deg_curve'] / 2)  # Normalize to 0-1
        
        # Simple heuristic: overcut works better with higher tire life and larger gaps
        probability = 0.5 - (0.2 * tire_advantage) + (0.2 * gap_factor) + (0.1 * deg_factor)
        probability = max(0.05, min(0.95, probability))  # Clamp between 0.05 and 0.95
        
        # Calculate confidence score
        confidence = self._get_confidence_score(probability)
        
        # Get recommended action
        recommendation = self._get_recommended_action(probability, "overcut")
        
        return {
            "success_probability": float(probability),
            "confidence_score": float(confidence),
            "recommended_action": recommendation,
            "factors": self.overcut_feature_importance
        }
    
    def train(self, training_data: pd.DataFrame, strategy_type: str):
        """Train the model using historical data"""
        # This would be implemented in a real system
        # For now, we'll just print a message
        print(f"Training {strategy_type} model with {len(training_data)} samples")
        
        # In a real implementation, we would:
        # 1. Preprocess the data
        # 2. Split into train/test sets
        # 3. Scale features
        # 4. Train the model
        # 5. Evaluate performance
        # 6. Update feature importances
        pass
    
    def save_models(self, model_path: str):
        """Save trained models to disk"""
        os.makedirs(model_path, exist_ok=True)
        joblib.dump(self.undercut_model, os.path.join(model_path, 'undercut_model.joblib'))
        joblib.dump(self.overcut_model, os.path.join(model_path, 'overcut_model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.joblib'))
        
        # Save feature importances
        with open(os.path.join(model_path, 'feature_importance.json'), 'w') as f:
            import json
            json.dump({
                'undercut': self.undercut_feature_importance,
                'overcut': self.overcut_feature_importance
            }, f)