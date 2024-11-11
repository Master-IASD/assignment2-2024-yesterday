import os
import subprocess
import json
from datetime import datetime
from tqdm import tqdm
from metrics.evaluate import MetricsEvaluator, save_evaluation_results, print_evaluation_results

def run_experiment(k_value: float, model_type: str = 'vanilla', base_dir: str = 'experiments') -> dict:
    """Run a single OBRS experiment with given K value"""
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'k_{k_value}_{timestamp}'
    experiment_dir = os.path.join(base_dir, experiment_name)
    samples_dir = os.path.join(experiment_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"\n=== Running experiment with K={k_value} ===")
    
    # Run generation with reduced batch size
    generate_cmd = [
        'python', 
        'generate.py',
        '--model', model_type,
        '--sampling_method', 'obrs',
        '--budget_K', str(k_value),
        '--output_dir', samples_dir,
        '--batch_size', '64'  # Reduced from 128 to 64
    ]
    
    print(f"Running command: {' '.join(generate_cmd)}")
    process = subprocess.run(generate_cmd, 
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           universal_newlines=True)
    
    if process.returncode != 0:
        print(f"Error in generation:\nstdout: {process.stdout}\nstderr: {process.stderr}")
        return None
    
    print("Generation complete. Running evaluation...")
    
    # Initialize evaluator with reduced batch size
    evaluator = MetricsEvaluator(real_data_path='data/MNIST', batch_size=64)  # Reduced from 128 to 64
    
    # Run evaluation
    try:
        results = evaluator.evaluate_samples(samples_dir)
        results['k_value'] = k_value
        
        # Save individual result
        result_path = os.path.join(experiment_dir, 'results.json')
        save_evaluation_results(results, result_path)
        print_evaluation_results(results)
        
        return results
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None

def run_all_experiments(k_values: list, model_type: str = 'vanilla') -> list:
    """Run experiments for multiple K values"""
    base_dir = os.path.join('results', f'obrs_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Starting experiments. Results will be saved in: {base_dir}")
    results = []
    
    for k in k_values:
        result = run_experiment(k, model_type, base_dir)
        if result:
            results.append(result)
    
    # Save and display summary
    if results:
        summary_path = os.path.join(base_dir, 'summary.json')
        save_evaluation_results(results, summary_path)
        
        print("\n=== Final Summary ===")
        print("K-value | FID Score | Precision | Recall")
        print("----------------------------------------")
        for result in results:
            print(f"{result['k_value']:7.1f} | {result['fid_score']:9.2f} | {result['precision']:9.4f} | {result['recall']:.4f}")
        
        # Find best K for each metric
        best_fid = min(results, key=lambda x: x['fid_score'])
        best_precision = max(results, key=lambda x: x['precision'])
        best_recall = max(results, key=lambda x: x['recall'])
        
        print("\nBest Results:")
        print(f"Best FID Score: K={best_fid['k_value']} (FID={best_fid['fid_score']:.2f})")
        print(f"Best Precision: K={best_precision['k_value']} (Precision={best_precision['precision']:.4f})")
        print(f"Best Recall: K={best_recall['k_value']} (Recall={best_recall['recall']:.4f})")
        
        print(f"\nDetailed results saved to: {summary_path}")
    
    return results

if __name__ == '__main__':
    # Define K values to test
    k_values = [2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    
    print("Starting OBRS K-value testing")
    print(f"Testing K values: {k_values}")
    
    results = run_all_experiments(k_values, model_type='vanilla')