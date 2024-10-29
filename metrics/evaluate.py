# # evaluate.py
# import torch
# import torchvision
# import os
# import argparse
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# import json
# from datetime import datetime
# import random
# import time
# import torch.nn.functional as F

# def load_samples(samples_dir, target_size=(28, 28), max_samples=None):
#     """Load, resize, convert to grayscale, and normalize generated samples from the directory."""
#     abs_path = os.path.abspath(samples_dir)
#     print(f"Looking for samples in: {abs_path}")
    
#     image_files = [f for f in sorted(os.listdir(samples_dir)) if f.endswith('.png')]
    
#     if max_samples and max_samples < len(image_files):
#         image_files = random.sample(image_files, max_samples)
#         print(f"Using {max_samples} randomly sampled images")
    
#     print(f"Loading {len(image_files)} PNG files...")
#     images = []
    
#     for filename in tqdm(image_files, desc="Loading images"):
#         img_path = os.path.join(samples_dir, filename)
#         img = torchvision.io.read_image(img_path).float() / 255.0  # Read image and normalize
        
#         # Convert to grayscale if it's in RGB format
#         if img.size(0) == 3:  # RGB format check
#             img = transforms.functional.rgb_to_grayscale(img)
        
#         # Resize to (28, 28) and normalize
#         img = F.interpolate(img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
#         img = (img - 0.5) / 0.5  # Normalize as in the real MNIST data
#         images.append(img)
    
#     images = torch.stack(images)
#     print(f"Generated images resized to {target_size} and stacked.")
#     return images

# def calculate_precision_recall(real_features, fake_features, k=3):
#     """Calculate precision and recall metrics."""
#     # Normalize features
#     real_features = F.normalize(real_features, dim=1)
#     fake_features = F.normalize(fake_features, dim=1)
    
#     # Calculate pairwise distances
#     print("Computing distances...")
#     real_distances = torch.cdist(real_features, real_features)
#     fake_distances = torch.cdist(fake_features, real_features)
    
#     # Find k-nearest neighbors
#     print("Finding nearest neighbors...")
#     real_nearest = torch.topk(real_distances, k=k+1, dim=1, largest=False)[0][:, 1:]
#     fake_nearest = torch.topk(fake_distances, k=k, dim=1, largest=False)[0]
    
#     # Calculate metrics
#     real_radius = torch.max(real_nearest, dim=1)[0]
#     precision = torch.mean((fake_nearest <= real_radius.unsqueeze(1)).float())
#     recall = torch.mean((fake_distances.min(dim=0)[0] <= real_radius).float())
    
#     return precision.item(), recall.item()


# def extract_features(images):
#     """Extract features by flattening images."""
#     features = images.view(images.size(0), -1)  # Flatten to (batch_size, num_features)
#     return features

# def main(args):
#     start_time = time.time()
#     print(f"Starting evaluation with {args.max_samples} samples")
    
#     os.makedirs(args.results_dir, exist_ok=True)
    
#     try:
#         # Load and resize generated samples
#         generated_images = load_samples(args.samples_dir, target_size=(28, 28), max_samples=args.max_samples)
        
#         # Load and resize real images
#         transform = transforms.Compose([
#             transforms.Resize((28, 28)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.5,), std=(0.5,))
#         ])
#         dataset = datasets.MNIST(root='../data/MNIST/', train=False, transform=transform, download=True)
#         if args.max_samples:
#             indices = random.sample(range(len(dataset)), args.max_samples)
#             dataset = Subset(dataset, indices)
        
#         real_images = torch.stack([dataset[i][0] for i in range(len(dataset))])
        
#         print("\nExtracting features...")
#         real_features = extract_features(real_images)
#         generated_features = extract_features(generated_images)
        
#         print(f"Real features size: {real_features.size()}")
#         print(f"Generated features size: {generated_features.size()}")
        
#         # Verify feature set dimension match
#         if real_features.size(1) != generated_features.size(1):
#             raise ValueError(f"Feature dimension mismatch: real features have {real_features.size(1)} columns, "
#                              f"while generated features have {generated_features.size(1)} columns.")

#         print("\nCalculating precision and recall...")
#         precision, recall = calculate_precision_recall(real_features, generated_features)
        
#         print(f"\nResults:")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
        
#         results = {
#             'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
#             'metrics': {
#                 'precision': float(precision),
#                 'recall': float(recall)
#             },
#             'config': {
#                 'samples_directory': args.samples_dir,
#                 'max_samples': args.max_samples,
#                 'computation_time': time.time() - start_time
#             }
#         }
        
#         result_file = os.path.join(args.results_dir, f'precision_recall_results_{results["timestamp"]}.json')
#         with open(result_file, 'w') as f:
#             json.dump(results, f, indent=4)
        
#         print(f"\nResults saved to {result_file}")
#         print(f"Total computation time: {(time.time() - start_time):.2f} seconds")
        
#     except Exception as e:
#         print(f"\nAn error occurred: {str(e)}")
#         raise

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Calculate precision and recall metrics.')
#     parser.add_argument('--samples_dir', type=str, required=True, help='Directory containing generated samples')
#     parser.add_argument('--batch_size', type=int, default=50, help='Batch size for loading samples')
#     parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to use')
#     parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save evaluation results')
    
#     args = parser.parse_args()
#     main(args)


import torch
from torch_fidelity import calculate_metrics
import os

if __name__ == '__main__':
    # Path to the generated samples
    generated_samples_path = 'samples'

    # Path to the real MNIST dataset (you need to have this dataset in a similar format)
    real_samples_path = 'data/MNIST/train'

    # Debugging: Print paths and list files
    print(f"Generated samples path: {generated_samples_path}")
    print(f"Real samples path: {real_samples_path}")

    # List files in the directories
    print(f"Files in generated samples path: {os.listdir(generated_samples_path)}")
    print(f"Files in real samples path: {os.listdir(real_samples_path)}")

    # Calculate metrics
    metrics_dict = calculate_metrics(
        input1=generated_samples_path,
        input2=real_samples_path,
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        prc=True,
        verbose=True,
        samples_find_deep=True  # Enable recursive search
    )

    print(metrics_dict)

    # Save the results to a file
    import json
    with open('metrics/evaluation_results.json', 'w') as f:
        json.dump(metrics_dict, f)
