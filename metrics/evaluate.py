import torch
import torchvision
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
from datetime import datetime
import random
import time
import torch.nn.functional as F

def load_real_samples(batch_size, max_samples=None):
    """Load real MNIST samples."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    dataset = datasets.MNIST(root='../data/MNIST/', train=False,
                           transform=transform, download=True)
    
    if max_samples and max_samples < len(dataset):
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, indices)
        print(f"Using {max_samples} randomly sampled real images")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def load_generated_samples(samples_dir, batch_size, max_samples=None):
    """Load generated samples from directory."""
    abs_path = os.path.abspath(samples_dir)
    print(f"Looking for samples in: {abs_path}")
    
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Samples directory not found: {abs_path}")
    
    image_files = [f for f in sorted(os.listdir(samples_dir)) if f.endswith('.png')]
    
    if not image_files:
        raise ValueError(f"No PNG files found in {abs_path}")
    
    if max_samples and max_samples < len(image_files):
        image_files = random.sample(image_files, max_samples)
        print(f"Using {max_samples} randomly sampled generated images")
    
    print(f"Loading {len(image_files)} PNG files...")
    
    # Load images in batches for better memory efficiency
    generated_images = []
    batch_size_load = 100  # Load images in smaller batches
    
    for i in tqdm(range(0, len(image_files), batch_size_load), desc="Loading image batches"):
        batch_files = image_files[i:i + batch_size_load]
        batch_images = []
        
        for filename in batch_files:
            img_path = os.path.join(samples_dir, filename)
            img = torchvision.io.read_image(img_path).float() / 255.0
            img = (img - 0.5) / 0.5  # Normalize
            batch_images.append(img)
        
        generated_images.extend(batch_images)
        torch.cuda.empty_cache()  # Clear GPU memory after each batch
    
    # Convert to tensor and create dataloader
    generated_images = torch.stack(generated_images)
    dataset = torch.utils.data.TensorDataset(generated_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def extract_features(dataloader, model, desc="Extracting features"):
    """Extract features from images using the model."""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.cuda()
            features = model(batch)
            features_list.append(features.cpu())
            torch.cuda.empty_cache()  # Clear GPU memory after each batch
    
    return torch.cat(features_list, dim=0)

def main(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Starting evaluation with {args.max_samples} samples per set")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    try:
        # Load data with progress bars
        real_loader = load_real_samples(args.batch_size, args.max_samples)
        generated_loader = load_generated_samples(args.samples_dir, args.batch_size, args.max_samples)
        
        # Initialize feature extractor
        print("\nInitializing InceptionV3 feature extractor...")
        from evaluations import InceptionV3Features, calculate_fid, calculate_precision_recall
        feature_extractor = InceptionV3Features().to(device)
        
        # Extract features with separate progress bars
        print("\nExtracting features...")
        real_features = extract_features(real_loader, feature_extractor, "Real images")
        generated_features = extract_features(generated_loader, feature_extractor, "Generated images")
        
        # Calculate metrics
        print("\nComputing metrics...")
        fid = calculate_fid(real_loader, generated_loader, args.batch_size)
        precision, recall = calculate_precision_recall(real_features, generated_features)
        
        # Save results
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'metrics': {
                'fid': float(fid),
                'precision': float(precision),
                'recall': float(recall)
            },
            'config': {
                'samples_directory': args.samples_dir,
                'batch_size': args.batch_size,
                'max_samples': args.max_samples,
                'computation_time': time.time() - start_time
            }
        }
        
        result_file = os.path.join(args.results_dir, f'evaluation_results_{results["timestamp"]}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\nResults:")
        print(f"FID Score: {fid:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"\nResults saved to {result_file}")
        print(f"Total computation time: {(time.time() - start_time)/60:.2f} minutes")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Cleaning up...")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        torch.cuda.empty_cache()
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generated samples.')
    parser.add_argument('--samples_dir', type=str, required=True,
                      help='Directory containing generated samples')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=1000,
                      help='Maximum number of samples to use for evaluation')
    parser.add_argument('--results_dir', type=str, default='./results',
                      help='Directory to save evaluation results')
    
    args = parser.parse_args()
    main(args)