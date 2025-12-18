"""
UAV Mission Prioritization DSS - Scenario Testing Module
Tests system performance under different scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from decision_engine import DecisionSupportSystem
import time

class ScenarioTester:
    """Systematic scenario testing for UAV mission prioritization DSS"""
    
    def __init__(self):
        self.scenarios = {
            'Default': {
                'urgency_score': 0.4521,
                'threat_level': 0.2645,
                'civilian_density': 0.0937,
                'weather_suitability': 0.0780,
                'uav_availability': 0.1117
            },
            'Urgency Focused': {
                'urgency_score': 0.60,
                'threat_level': 0.20,
                'civilian_density': 0.10,
                'weather_suitability': 0.05,
                'uav_availability': 0.05
            },
            'Threat Focused': {
                'urgency_score': 0.25,
                'threat_level': 0.50,
                'civilian_density': 0.15,
                'weather_suitability': 0.05,
                'uav_availability': 0.05
            },
            'Civilian Safety Focused': {
                'urgency_score': 0.30,
                'threat_level': 0.20,
                'civilian_density': 0.35,
                'weather_suitability': 0.10,
                'uav_availability': 0.05
            },
            'Balanced': {
                'urgency_score': 0.20,
                'threat_level': 0.20,
                'civilian_density': 0.20,
                'weather_suitability': 0.20,
                'uav_availability': 0.20
            }
        }
    
    def run_all_scenarios(self, missions_df, uav_df, weather_df, sample_size=1000):
        """Run all scenarios and compare results"""
        
        print("\n" + "="*70)
        print("SCENARIO TESTING STARTING")
        print("="*70 + "\n")
        
        # Get sample (for testing speed)
        if len(missions_df) > sample_size:
            test_missions = missions_df.sample(n=sample_size, random_state=42)
            print(f"Tested {sample_size} mission samples\n")
        else:
            test_missions = missions_df
        
        results = {}
        
        for scenario_name, weights in self.scenarios.items():
            print(f"Scenario: {scenario_name}")
            print(f"Weights: {weights}")

            start_time = time.time()
            
            # Run DSS
            dss = DecisionSupportSystem()
            dss.load_data(test_missions, uav_df, weather_df)
            scenario_results = dss.run_analysis(custom_weights=weights)
            
            elapsed = time.time() - start_time
            
            # Save results
            results[scenario_name] = {
                'results': scenario_results,
                'weights': weights,
                'elapsed_time': elapsed
            }
            
            print(f"Completed ({elapsed:.2f} seconds)\n")
        
        print("="*70)
        print("ALL SCENARIOS COMPLETED")
        print("="*70 + "\n")
        
        return results
    
    def compare_top_missions(self, results, top_n=10):
        """Compare top priority missions across scenarios"""
        
        print("\n" + "="*70)
        print(f"TOP {top_n} MISSION COMPARISON")
        print("="*70 + "\n")
        
        comparison = {}
        
        for scenario_name, data in results.items():
            top_missions = data['results'].nsmallest(top_n, 'rank')['mission_id'].tolist()
            comparison[scenario_name] = top_missions
        
        # Create DataFrame 
        comparison_df = pd.DataFrame(comparison)
        
        print(comparison_df)
        print("\n")
        
        # Find common missions
        all_scenarios = list(results.keys())
        common_missions = set(comparison[all_scenarios[0]])
        
        for scenario in all_scenarios[1:]:
            common_missions = common_missions.intersection(set(comparison[scenario]))
        
        print(f" All missions common across all scenarios: {len(common_missions)}")
        print(f"{list(common_missions)}\n")
        
        return comparison_df
    
    def analyze_ranking_changes(self, results, mission_ids=None):
        """Analyze ranking changes for specified missions across scenarios"""
        
        if mission_ids is None:
            # Get top 20 missions from the first scenario
            first_scenario = list(results.values())[0]['results']
            mission_ids = first_scenario.nsmallest(20, 'rank')['mission_id'].tolist()
        
        print("\n" + "="*70)
        print("MISSION RANKING CHANGE ANALYSIS")
        print("="*70 + "\n")
        
        ranking_data = []
        
        for mission_id in mission_ids:
            mission_ranks = {}
            
            for scenario_name, data in results.items():
                rank = data['results'][data['results']['mission_id'] == mission_id]['rank'].values
                if len(rank) > 0:
                    mission_ranks[scenario_name] = int(rank[0])
            
            ranking_data.append({
                'mission_id': mission_id,
                **mission_ranks
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        print(ranking_df)
        print("\n")
        
        # Calculate variability statistics
        ranking_numeric = ranking_df.drop('mission_id', axis=1)
        ranking_df['std'] = ranking_numeric.std(axis=1)
        ranking_df['range'] = ranking_numeric.max(axis=1) - ranking_numeric.min(axis=1)
        
        print("Variability Statistics:")
        print(f"   Average standard deviation: {ranking_df['std'].mean():.2f}")
        print(f"   Average ranking difference: {ranking_df['range'].mean():.2f}")
        print(f"   Most variable mission: {ranking_df.loc[ranking_df['std'].idxmax(), 'mission_id']}")
        print(f"   Most stable mission: {ranking_df.loc[ranking_df['std'].idxmin(), 'mission_id']}\n")
        
        return ranking_df
    
    def visualize_scenario_comparison(self, results, save_path='results/'):
        """Visualize scenario comparison results"""
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("\n" + "="*70)
        print("VISUALIZATIONS BEING CREATED")
        print("="*70 + "\n")
        
        # 1. Criteria weights comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        weights_data = []
        for scenario_name, data in results.items():
            for criterion, weight in data['weights'].items():
                weights_data.append({
                    'Scenario': scenario_name,
                    'Criterion': criterion,
                    'Weight': weight
                })
        
        weights_df = pd.DataFrame(weights_data)
        weights_pivot = weights_df.pivot(index='Criterion', columns='Scenario', values='Weight')
        
        weights_pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Criteria Weights Comparison Across Scenarios', fontsize=14, fontweight='bold')
        ax.set_xlabel('Criteria', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{save_path}scenario_weights.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ {save_path}scenario_weights.png")
        plt.close()
        
        # 2. Top 10 missions' TOPSIS scores
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, (scenario_name, data) in enumerate(results.items()):
            top_10 = data['results'].nsmallest(10, 'rank')
            ax.barh([f"{m} ({scenario_name})" for m in top_10['mission_id']], 
                   top_10['topsis_score'],
                   label=scenario_name,
                   alpha=0.7)
        
        ax.set_title('Top 10 Missions TOPSIS Scores by Scenario', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('TOPSIS Score', fontsize=12)
        ax.legend(title='Scenario')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{save_path}scenario_top10.png', dpi=300, bbox_inches='tight')
        print(f"{save_path}scenario_top10.png")
        plt.close()
        
        # 3. Performance metrics comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Processing times
        scenario_names = list(results.keys())
        elapsed_times = [results[s]['elapsed_time'] for s in scenario_names]
        
        axes[0].bar(scenario_names, elapsed_times, color='skyblue', edgecolor='navy')
        axes[0].set_title('Scenario Processing Times', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Süre (saniye)', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Average TOPSIS scores
        avg_scores = [results[s]['results']['topsis_score'].mean() for s in scenario_names]
        
        axes[1].bar(scenario_names, avg_scores, color='lightcoral', edgecolor='darkred')
        axes[1].set_title('Average TOPSIS Scores', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Score', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}scenario_metrics.png', dpi=300, bbox_inches='tight')
        print(f"   ✓ {save_path}scenario_metrics.png")
        plt.close()
        
        print("\n Visualizations saved!\n")
    
    def generate_report(self, results, save_path='results/scenario_report.txt'):
        """Generate detailed scenario test report"""
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("IHA MISSION PRIORITIZATION - SCENARIO TEST REPORT\n")
            f.write("="*70 + "\n\n")
            
            for scenario_name, data in results.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"SCENARIO: {scenario_name}\n")
                f.write(f"{'='*70}\n\n")
                
                f.write("Criteria Weights:\n")
                for criterion, weight in data['weights'].items():
                    f.write(f"  • {criterion}: {weight:.4f}\n")
                
                f.write(f"\nProcessing Time: {data['elapsed_time']:.2f} seconds\n")
                
                results_df = data['results']
                
                f.write("\nStatistics:\n")
                f.write(f"  • Total missions: {len(results_df)}\n")
                f.write(f"  • Average TOPSIS score: {results_df['topsis_score'].mean():.4f}\n")
                f.write(f"  • Standard deviation: {results_df['topsis_score'].std():.4f}\n")
                f.write(f"  • Min score: {results_df['topsis_score'].min():.4f}\n")
                f.write(f"  • Max score: {results_df['topsis_score'].max():.4f}\n")
                
                f.write("\nTop 10 Missions:\n")
                top_10 = results_df.nsmallest(10, 'rank')[['mission_id', 'mission_type', 
                                                            'urgency_level', 'topsis_score', 'rank']]
                f.write(top_10.to_string(index=False))
                f.write("\n")
        
        print(f"Detailed report saved: {save_path}\n")


def main():
    """Run scenario tests for UAV mission prioritization DSS"""
    
    print("\n" + "="*70)
    print("UAV MISSION PRIORITIZATION DSS - SCENARIO TESTING")
    print("="*70)
    
    # Load data
    print("\n Loading data...")
    missions = pd.read_csv('data/missions.csv')
    uav_fleet = pd.read_csv('data/uav_fleet.csv')
    weather = pd.read_csv('data/weather.csv')
    print(f"{len(missions):,} missions loaded")
    
    # Create Tester
    tester = ScenarioTester()
    
    # Run scenarios (test with 1000 missions)
    results = tester.run_all_scenarios(missions, uav_fleet, weather, sample_size=1000)
    
    # Comparisons
    tester.compare_top_missions(results, top_n=10)
    tester.analyze_ranking_changes(results)
    
    # Visualizations
    tester.visualize_scenario_comparison(results)
    
    # Generate report
    tester.generate_report(results)
    
    print("="*70)
    print("ALL TESTS COMPLETED!")
    print("Results saved in 'results/' folder")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()