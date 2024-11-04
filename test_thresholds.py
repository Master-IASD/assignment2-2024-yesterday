import subprocess
import time
import os

def calculate_metrics(threshold, generation_time):
    # Initialize metrics with default values
    fid = float('inf')
    precision = 0.0
    recall = 0.0
    
    # Calculate FID
    print(f"\nCalculating metrics for threshold {threshold}...")
    cmd = "python -m metrics.evaluate --real_data_path data/MNIST --generated_data_path samples"
    process = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    # Print full output for debugging
    print("Raw metrics output:")
    print(process.stdout)
    print("Error output (if any):")
    print(process.stderr)
    
    # Parse the metrics from output
    if process.stdout:
        metrics_lines = process.stdout.strip().split('\n')
        for line in metrics_lines:
            line = line.lower()  # Convert to lowercase for easier matching
            try:
                if 'fid' in line:
                    fid = float(line.split(':')[-1].strip())
                elif 'precision' in line:
                    precision = float(line.split(':')[-1].strip())
                elif 'recall' in line:
                    recall = float(line.split(':')[-1].strip())
            except ValueError as e:
                print(f"Error parsing line: {line}")
                print(f"Error details: {e}")
                continue
    
    result = {
        'threshold': threshold,
        'generation_time': generation_time,
        'fid': fid,
        'precision': precision,
        'recall': recall
    }
    
    # Print individual result
    print(f"\nResults for threshold {threshold}:")
    print(f"FID: {fid}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Generation time: {generation_time:.2f}s")
    
    return result

def test_threshold(threshold):
    # Command to run the generator with specific threshold
    cmd = f"python generate.py --model wgan --rejection_sampling --threshold {threshold} --batch_size 2048"
    
    # Run the generation process
    print(f"\nGenerating samples with threshold {threshold}...")
    start_time = time.time()
    process = subprocess.run(cmd.split(), capture_output=True, text=True)
    generation_time = time.time() - start_time
    
    print(f"Generation time: {generation_time:.2f} seconds")
    print("Generator Output:")
    print(process.stdout)
    
    # Calculate and return all metrics
    return calculate_metrics(threshold, generation_time)

# Test range of thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
results = []

try:
    for threshold in thresholds:
        print(f"\n{'='*50}")
        print(f"Testing threshold: {threshold}")
        print(f"{'='*50}")
        
        result = test_threshold(threshold)
        results.append(result)
        
        # Save intermediate results to file
        with open('drs_results.txt', 'a') as f:
            f.write(f"\nThreshold: {threshold}\n")
            f.write(f"FID: {result['fid']}\n")
            f.write(f"Precision: {result['precision']}\n")
            f.write(f"Recall: {result['recall']}\n")
            f.write(f"Generation time: {result['generation_time']:.2f}s\n")
            f.write("-" * 30 + "\n")

except Exception as e:
    print(f"Error during testing: {e}")
    # Save any results we got before the error
    if results:
        print("\nPartial results before error:")
        for result in results:
            print(f"Threshold {result['threshold']}: FID={result['fid']}, "
                  f"Precision={result['precision']}, Recall={result['recall']}")

finally:
    if results:
        # Print final summary table for whatever results we have
        print("\n" + "="*70)
        print("Results Summary:")
        print("-"*70)
        print(f"{'Threshold':^10} | {'FID':^12} | {'Precision':^10} | {'Recall':^10} | {'Gen Time':^10}")
        print("-"*70)
        for result in results:
            print(f"{result['threshold']:^10.1f} | {result['fid']:^12.2f} | "
                  f"{result['precision']:^10.3f} | {result['recall']:^10.3f} | "
                  f"{result['generation_time']:^10.1f}")
        print("-"*70)

        # Find best results among what we have
        best_fid = min(results, key=lambda x: x['fid'])
        best_precision = max(results, key=lambda x: x['precision'])
        best_recall = max(results, key=lambda x: x['recall'])

        print("\nBest Results:")
        print(f"Best FID: {best_fid['fid']:.2f} (threshold = {best_fid['threshold']})")
        print(f"Best Precision: {best_precision['precision']:.3f} "
              f"(threshold = {best_precision['threshold']})")
        print(f"Best Recall: {best_recall['recall']:.3f} "
              f"(threshold = {best_recall['threshold']})")