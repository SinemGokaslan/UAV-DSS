"""
UAV Mission Prioritization DSS - Decision Engine
AHP (Analytic Hierarchy Process) + TOPSIS 
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AHPEngine:
    """
    AHP (Analytic Hierarchy Process) - Criterion Weighting
    Determinates weights for decision criteria based on pairwise comparisons
    """
    
    def __init__(self):
        self.criteria = [
            'urgency_score',      # Mission Urgency
            'threat_level',       # Threat Level
            'civilian_density',   # Civilian Density
            'weather_suitability',# Weather Suitability
            'uav_availability'    # UAV Availability
        ]
        
        # Comparison matrix (expert opinion simulation)
        # 1: Equal, 3: Moderate, 5: Strongly Important, 7: Very Strongly Important, 9: Absolutely Important
        self.comparison_matrix = None
        self.weights = None
        
    def create_comparison_matrix(self, expert_input=None):
        """
        Create binary comparison matrix 
        expert_input: User-defined priorities (optional)
        """
        n = len(self.criteria)
        
        if expert_input is None:
            # Default expert input (logical values for mission scenario)
            matrix = np.array([
                [1,   3,   5,   7,   5],  # Urgency(most important)
                [1/3, 1,   3,   5,   3],  # Threat Level
                [1/5, 1/3, 1,   3,   1],  # Civilian Density
                [1/7, 1/5, 1/3, 1,   1/3],# Weather Suitability
                [1/5, 1/3, 1,   3,   1]   # UAV Availability
            ])
        else:
            matrix = expert_input
        
        self.comparison_matrix = matrix
        return matrix
    
    def calculate_weights(self):
        """
         Calculate weights (Eigen Vector Method)
        """
        if self.comparison_matrix is None:
            self.create_comparison_matrix()
        
        # Column sums
        col_sums = self.comparison_matrix.sum(axis=0)
        
        # Normalized matrix
        normalized = self.comparison_matrix / col_sums
        
        # Weights calculation (row averages)
        weights = normalized.mean(axis=1)
        
        self.weights = weights
        return weights
    
    def calculate_consistency_ratio(self):
        """
        Calculate consistency ratio (Must be CR < 0.10)
        """
        if self.comparison_matrix is None or self.weights is None:
            raise ValueError("First create the matrix and calculate weights")
        
        n = len(self.criteria)
        
        # Calculate λ_max
        weighted_sum = self.comparison_matrix @ self.weights
        lambda_max = (weighted_sum / self.weights).mean()
        
        # Consistency Index (CI)
        ci = (lambda_max - n) / (n - 1)
        
        # Random Index (RI) - From Saaty table
        ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 
                   7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_table.get(n, 1.49)
        
        # Consistency Ratio (CR)
        cr = ci / ri if ri != 0 else 0
        
        return {
            'lambda_max': lambda_max,
            'ci': ci,
            'ri': ri,
            'cr': cr,
            'is_consistent': cr < 0.10
        }
    
    def get_weights_dict(self):
        """ Return weights as dictionary """
        if self.weights is None:
            self.calculate_weights()
        
        return dict(zip(self.criteria, self.weights))


class TOPSISEngine:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    Multi-Criteria Decision Making method to rank alternatives
    """
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.criteria = list(weights.keys())
        
    def normalize_matrix(self, df: pd.DataFrame):
        """
        Normalize decision matrix (Vector normalization)
        """
        normalized = df[self.criteria].copy()
        
        for col in self.criteria:
            vector_length = np.sqrt((df[col] ** 2).sum())
            if vector_length != 0:
                normalized[col] = df[col] / vector_length
            else:
                normalized[col] = 0
        
        return normalized
    
    def apply_weights(self, normalized_df: pd.DataFrame):
        """
        NApply weights to normalized matrix
        """
        weighted = normalized_df.copy()
        
        for col in self.criteria:
            weighted[col] = normalized_df[col] * self.weights[col]
        
        return weighted
    
    def find_ideal_solutions(self, weighted_df: pd.DataFrame, benefit_criteria: List[str]):
        """
        Ideal and negative-ideal solutions
        benefit_criteria: Criteria to be maximized (higher=better)
        cost_criteria: Criteria to be minimized (lower=better)
        """
        ideal_positive = {}
        ideal_negative = {}
        
        for col in self.criteria:
            if col in benefit_criteria:
                # Benefit criterion: maximum = ideal
                ideal_positive[col] = weighted_df[col].max()
                ideal_negative[col] = weighted_df[col].min()
            else:
                # Cost criterion: minimum = ideal
                ideal_positive[col] = weighted_df[col].min()
                ideal_negative[col] = weighted_df[col].max()
        
        return ideal_positive, ideal_negative
    
    def calculate_distances(self, weighted_df: pd.DataFrame, 
                          ideal_positive: Dict, ideal_negative: Dict):
        """
        Calculate distances to ideal and negative-ideal solutions
        """
        n = len(weighted_df)
        
        # Distance to ideal solution (S+)
        dist_positive = np.zeros(n)
        # Distance to ideal negative (S-)
        dist_negative = np.zeros(n)
        
        for i in range(n):
            sum_pos = 0
            sum_neg = 0
            
            for col in self.criteria:
                sum_pos += (weighted_df.iloc[i][col] - ideal_positive[col]) ** 2
                sum_neg += (weighted_df.iloc[i][col] - ideal_negative[col]) ** 2
            
            dist_positive[i] = np.sqrt(sum_pos)
            dist_negative[i] = np.sqrt(sum_neg)
        
        return dist_positive, dist_negative
    
    def calculate_scores(self, dist_positive: np.ndarray, dist_negative: np.ndarray):
        """
        Calculate TOPSIS scores (range 0-1, 1=best)
        """
        # Check for division by zero
        total_dist = dist_positive + dist_negative
        scores = np.where(total_dist != 0, 
                         dist_negative / total_dist, 
                         0)
        
        return scores
    
    def rank_alternatives(self, df: pd.DataFrame, benefit_criteria: List[str] = None):
        """
        Rank alternatives using TOPSIS
        benefit_criteria: List of criteria to be maximized"""
        if benefit_criteria is None:
            # Default: Urgency, threat, UAV availability = benefit (higher=better)
            # Civil density = cost (lower=better)
            benefit_criteria = ['urgency_score', 'threat_level', 'uav_availability', 
                              'weather_suitability']
        
        # 1. Normalize 
        normalized = self.normalize_matrix(df)
        
        # 2. Apply weights
        weighted = self.apply_weights(normalized)
        
        # 3. Ideal solutions
        ideal_pos, ideal_neg = self.find_ideal_solutions(weighted, benefit_criteria)
        
        # 4. Calculate distance
        dist_pos, dist_neg = self.calculate_distances(weighted, ideal_pos, ideal_neg)
        
        # 5. Calculate scores
        scores = self.calculate_scores(dist_pos, dist_neg)
        
        # Add results to DataFrame
        result = df.copy()
        result['topsis_score'] = scores
        result['rank'] = result['topsis_score'].rank(ascending=False, method='min').astype(int)
        
        # Rank 
        result = result.sort_values('rank')
        
        return result


class DecisionSupportSystem:
    """
    Main Decision Support System
    Integrates AHP and TOPSIS
    """
    
    def __init__(self):
        self.ahp = AHPEngine()
        self.topsis = None
        self.mission_data = None
        self.uav_data = None
        self.weather_data = None
        
    def load_data(self, missions_df: pd.DataFrame, uav_df: pd.DataFrame, 
                  weather_df: pd.DataFrame = None):
        """Load datasets"""
        self.mission_data = missions_df.copy()
        self.uav_data = uav_df.copy()
        self.weather_data = weather_df
        
    def prepare_decision_matrix(self):
        """
        Prepare decision matrix (transform missions into evaluable form)
        """
        if self.mission_data is None:
            raise ValueError("Load data first: load_data()")
        
        df = self.mission_data.copy()
        
        # 1. Add weather suitability score
        if self.weather_data is not None:
            # Weather scoring (based on temperature and wind)
            weather_scores = []
            for _, mission in df.iterrows():
                # Find nearest weather location (simplified)
                if len(self.weather_data) > 0:
                    weather = self.weather_data.iloc[np.random.randint(0, len(self.weather_data))]
                    
                    # Calculate weather suitability (0-10 scale)
                    temp_score = 10 if 10 <= weather['temperature'] <= 30 else 5
                    wind_score = 10 - min(weather['wind_speed'] / 4, 10)  # Less wind = better
                    
                    weather_score = (temp_score + wind_score) / 2
                else:
                    weather_score = 7.0  # Default
                
                weather_scores.append(weather_score)
            
            df['weather_suitability'] = weather_scores
        else:
            # Random weather suitability if no data
            df['weather_suitability'] = np.random.uniform(5, 10, len(df))
        
        # 2. Add UAV availability score
        uav_availability_scores = []
        for _, mission in df.iterrows():
            # Number and status of available UAVs
            available_uavs = self.uav_data[
                (self.uav_data['status'] == 'Ready') & 
                (self.uav_data['fuel_level'] > 50) &
                (self.uav_data['sensor_type'] == mission['required_sensor'])
            ]
            
            if len(available_uavs) > 0:
                # Score based on fuel level
                avg_fuel = available_uavs['fuel_level'].mean()
                uav_score = min((avg_fuel / 100) * 10, 10)
            else:
                # No suitable UAVs but other UAVs are available
                all_available = self.uav_data[self.uav_data['status'] == 'Ready']
                if len(all_available) > 0:
                    uav_score = 5.0  # Medium level
                else:
                    uav_score = 2.0  # Low
            
            uav_availability_scores.append(uav_score)
        
        df['uav_availability'] = uav_availability_scores
        
        return df
    
    def run_analysis(self, custom_weights=None):
        """
        Run the full decision analysis
        """
        print("\n" + "="*70)
        print("DECISION SUPPORT SYSTEM ANALYSIS")
        print("="*70 + "\n")
        
        # 1. Calculate weights with AHP
        print("AHP - Criteria Weighting")
        if custom_weights is None:
            self.ahp.create_comparison_matrix()
            weights = self.ahp.calculate_weights()
            consistency = self.ahp.calculate_consistency_ratio()
            
            print(f" Weights Calculated")
            print(f" Consistency Rate (CR): {consistency['cr']:.4f}", end="")
            if consistency['is_consistent']:
                print("Consistent")
            else:
                print(" Inconsistent - (CR > 0.10)")
            
            weights_dict = self.ahp.get_weights_dict()
        else:
            weights_dict = custom_weights
            print(f" Using custom weights")
        
        print("\n  Criteria Weights:")
        for criterion, weight in weights_dict.items():
            print(f"      • {criterion}: {weight:.4f}")
        
        # 2. Prepare decision matrix
        print("\n Prepare Decision Matrix")
        decision_matrix = self.prepare_decision_matrix()
        print(f" {len(decision_matrix)} mission being evulated")
        
        # 3. Rank by TOPSIS
        print("\n  TOPSIS - Mission Prioritization")
        self.topsis = TOPSISEngine(weights_dict)
        results = self.topsis.rank_alternatives(decision_matrix)
        print(f" Missions ranked")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED")
        print("="*70 + "\n")
        
        return results
    
    def get_top_priorities(self, results: pd.DataFrame, n: int = 5):
        """Get top priority missions"""
        top = results.nsmallest(n, 'rank')
        
        return top[[
            'mission_id', 'mission_type', 'urgency_level', 
            'threat_level', 'civilian_density', 'topsis_score', 'rank'
        ]]
    
    def save_results(self, results: pd.DataFrame, filename: str = 'data/mission_priorities.csv'):
        """Save results"""
        results.to_csv(filename, index=False)
        print(f"Results saved: {filename}")


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_decision_engine():
    """Test the decision engine"""
    print("\n" + "="*70)
    print("DECISION ENGINE TEST")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    try:
        missions = pd.read_csv('data/missions.csv')
        uav_fleet = pd.read_csv('data/uav_fleet.csv')
        weather = pd.read_csv('data/weather.csv')
        print(f" {len(missions)} mission loaded")
        print(f" {len(uav_fleet)} UAVs loaded")
        print(f" {len(weather)} weather records loaded")
    except FileNotFoundError:
        print(" File Not Found Error!")
        print(" Firstly, run 'python src/data_collection.py'")
        return
    
    # Create Decision Support System
    dss = DecisionSupportSystem()
    dss.load_data(missions, uav_fleet, weather)
    
    # Run analysis
    results = dss.run_analysis()
    
    # Top priority missions
    print("TOP 5 PRIORITY MISSIONS:")
    print("="*70)
    top_5 = dss.get_top_priorities(results, n=5)
    print(top_5.to_string(index=False))
    
    # Save results
    print("\n" + "="*70)
    dss.save_results(results)
    
    print("\n Test completed!")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    results = test_decision_engine()