# import torch
# import torchvision
# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from pytorch_fid import fid_score
# import os
# import argparse
# from tqdm import tqdm
# import sys

# # Fix the path to the improved precision and recall package
# precision_recall_path = '/content/assignment2-2024-yesterday/improved-precision-and-recall-metric-pytorch'
# sys.path.append(precision_recall_path)

# try:
#     import improved_precision_recall
#     from improved_precision_recall import IPR, compute_metric, Manifold
# except ImportError as e:
#     print(f"Error importing module: {e}")
#     print("\nPlease ensure you have installed the package correctly:")
#     print("1. git clone https://github.com/youngjung/improved-precision-and-recall-metric-pytorch")
#     print("2. cd improved-precision-and-recall-metric-pytorch")
#     print("3. pip install -e .")
#     print("4. cd ..")
#     sys.exit(1)

# class GANEvaluator:
#     def __init__(self, real_data_path, generated_data_path, batch_size=64):
#         self.real_data_path = real_data_path
#         self.generated_data_path = generated_data_path
#         self.batch_size = batch_size
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # MNIST specific transforms
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#             transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Convert to 3 channels for FID
#         ])

#         self.ipr = IPR(batch_size=self.batch_size, k=3, num_samples=10000)

#     def compute_precision_recall(self):
#         """Compute precision and recall metrics"""
#         # Load real and generated images
#         real_loader = self.load_mnist_dataset()
#         real_images = []

#         # Collect real images
#         for imgs, _ in tqdm(real_loader, desc="Processing real images"):
#             real_images.append(imgs)
#         real_images = torch.cat(real_images, dim=0)

#         # Load and process generated images
#         generated_imgs = self.load_generated_images()

#         # Compute precision and recall using IPR
#         self.ipr.compute_manifold_ref(real_images)
#         precision, recall = self.ipr.precision_and_recall(generated_imgs)

#         return precision, recall

#     def load_mnist_dataset(self):
#         """Load MNIST dataset and convert to RGB format"""
#         dataset = datasets.MNIST(
#             root=self.real_data_path,
#             train=False,  # Use test set for evaluation
#             transform=self.transform,
#             download=True
#         )
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

#     def load_generated_images(self):
#         """Load generated images from directory"""
#         generated_images = []
#         files = sorted(os.listdir(self.generated_data_path))
#         for file in tqdm(files, desc="Loading generated images"):
#             if file.endswith('.png'):
#                 img = torchvision.io.read_image(os.path.join(self.generated_data_path, file))
#                 img = img.float() / 255.0  # Normalize to [0,1]
#                 img = (img - 0.5) / 0.5    # Normalize to [-1,1]
#                 # Convert to 3 channels if grayscale
#                 if img.shape[0] == 1:
#                     img = img.repeat(3, 1, 1)
#                 generated_images.append(img)
#         return torch.stack(generated_images)

#     def compute_fid(self):
#         """Compute FID score between real and generated images"""
#         # Create temporary directories for FID computation
#         real_tmp_dir = "tmp_real_fid"
#         gen_tmp_dir = "tmp_gen_fid"
#         os.makedirs(real_tmp_dir, exist_ok=True)
#         os.makedirs(gen_tmp_dir, exist_ok=True)

#         try:
#             # Save real images
#             loader = self.load_mnist_dataset()
#             for i, (imgs, _) in enumerate(tqdm(loader, desc="Preparing real images")):
#                 for j, img in enumerate(imgs):
#                     torchvision.utils.save_image(
#                         img,
#                         f"{real_tmp_dir}/{i}_{j}.png",
#                         normalize=True
#                     )

#             # Load and prepare generated images
#             generated_imgs = self.load_generated_images()
#             for i, img in enumerate(tqdm(generated_imgs, desc="Preparing generated images")):
#                 torchvision.utils.save_image(
#                     img,
#                     f"{gen_tmp_dir}/{i}.png",
#                     normalize=True
#                 )

#             # Compute FID score
#             fid = fid_score.calculate_fid_given_paths(
#                 [real_tmp_dir, gen_tmp_dir],
#                 batch_size=self.batch_size,
#                 device=self.device,
#                 dims=2048
#             )

#             return fid

#         finally:
#             # Cleanup temporary directories
#             import shutil
#             shutil.rmtree(real_tmp_dir, ignore_errors=True)
#             shutil.rmtree(gen_tmp_dir, ignore_errors=True)

#     def evaluate(self):
#         """Run all evaluation metrics and return results"""
#         print("Computing FID score...")
#         fid = self.compute_fid()

#         print("Computing Precision-Recall metrics...")
#         precision, recall = self.compute_precision_recall()

#         results = {
#             'fid': fid,
#             'precision': precision,
#             'recall': recall
#         }

#         return results

# def main():
#     parser = argparse.ArgumentParser(description='Evaluate GAN results on MNIST')
#     parser.add_argument('--real_data_path', type=str, default='data/MNIST',
#                       help='Path to real MNIST data')
#     parser.add_argument('--generated_data_path', type=str, default='samples',
#                       help='Path to generated samples')
#     parser.add_argument('--batch_size', type=int, default=64,
#                       help='Batch size for evaluation')
#     parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
#                       help='File to save evaluation results')

#     args = parser.parse_args()

#     evaluator = GANEvaluator(
#         args.real_data_path,
#         args.generated_data_path,
#         args.batch_size
#     )

#     results = evaluator.evaluate()

#     # Print and save results
#     print("\nEvaluation Results:")
#     print(f"FID Score: {results['fid']:.2f}")
#     print(f"Precision: {results['precision']:.4f}")
#     print(f"Recall: {results['recall']:.4f}")

#     with open(args.output_file, 'w') as f:
#         for metric, value in results.items():
#             f.write(f"{metric}: {value}\n")

# if __name__ == '__main__':
#     main()

import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_fid import fid_score
import os
from tqdm import tqdm
import sys
from typing import Dict, Tuple, List
import json
from datetime import datetime

# Add precision-recall metric path
PRECISION_RECALL_PATH = os.path.join(os.path.dirname(__file__), '../improved-precision-and-recall-metric-pytorch')
sys.path.append(PRECISION_RECALL_PATH)

try:
    from improved_precision_recall import IPR
except ImportError as e:
    print(f"Error importing IPR module: {e}")
    print("\nPlease install the improved-precision-and-recall package:")
    print("1. git clone https://github.com/youngjung/improved-precision-and-recall-metric-pytorch")
    print("2. cd improved-precision-and-recall-metric-pytorch")
    print("3. pip install -e .")
    sys.exit(1)

class MetricsEvaluator:
    """Class to handle all metrics evaluation for generated images"""
    
    def __init__(self, real_data_path: str, batch_size: int = 64):
        self.real_data_path = real_data_path
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # MNIST specific transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Convert to 3 channels
        ])
        
        # Initialize IPR metric
        self.ipr = IPR(batch_size=self.batch_size, k=3, num_samples=10000)
        
        # Cache for real images manifold
        self._real_manifold = None
    
    def load_mnist_dataset(self) -> DataLoader:
        """Load and prepare MNIST dataset"""
        dataset = datasets.MNIST(
            root=self.real_data_path,
            train=False,
            transform=self.transform,
            download=True
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def load_generated_images(self, generated_path: str) -> torch.Tensor:
        """Load and preprocess generated images"""
        generated_images = []
        files = sorted(os.listdir(generated_path))
        
        for file in tqdm(files, desc="Loading generated images"):
            if file.endswith('.png'):
                img_path = os.path.join(generated_path, file)
                img = torchvision.io.read_image(img_path)
                img = img.float() / 255.0  # Normalize to [0,1]
                img = (img - 0.5) / 0.5    # Normalize to [-1,1]
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                generated_images.append(img)
                
        return torch.stack(generated_images)
    
    def compute_fid(self, generated_path: str) -> float:
        """Compute FID score between real and generated images"""
        real_tmp_dir = "tmp_real_fid"
        gen_tmp_dir = "tmp_gen_fid"
        os.makedirs(real_tmp_dir, exist_ok=True)
        os.makedirs(gen_tmp_dir, exist_ok=True)
        
        try:
            # Save real images
            loader = self.load_mnist_dataset()
            for i, (imgs, _) in enumerate(tqdm(loader, desc="Preparing real images")):
                for j, img in enumerate(imgs):
                    torchvision.utils.save_image(
                        img,
                        f"{real_tmp_dir}/{i}_{j}.png",
                        normalize=True
                    )
            
            # Process generated images
            generated_imgs = self.load_generated_images(generated_path)
            for i, img in enumerate(tqdm(generated_imgs, desc="Preparing generated images")):
                torchvision.utils.save_image(
                    img,
                    f"{gen_tmp_dir}/{i}.png",
                    normalize=True
                )
            
            # Calculate FID
            fid = fid_score.calculate_fid_given_paths(
                [real_tmp_dir, gen_tmp_dir],
                batch_size=self.batch_size,
                device=self.device,
                dims=2048
            )
            
            return fid
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(real_tmp_dir, ignore_errors=True)
            shutil.rmtree(gen_tmp_dir, ignore_errors=True)
    
    def compute_precision_recall(self, generated_path: str) -> Tuple[float, float]:
        """Compute precision and recall metrics"""
        # Initialize real manifold if not already done
        if self._real_manifold is None:
            real_loader = self.load_mnist_dataset()
            real_images = []
            for imgs, _ in tqdm(real_loader, desc="Processing real images"):
                real_images.append(imgs)
            real_images = torch.cat(real_images, dim=0)
            self.ipr.compute_manifold_ref(real_images)
        
        # Load and evaluate generated images
        generated_imgs = self.load_generated_images(generated_path)
        precision, recall = self.ipr.precision_and_recall(generated_imgs)
        
        return precision, recall
    
    def evaluate_samples(self, generated_path: str) -> Dict[str, float]:
        """Evaluate all metrics for a set of generated samples"""
        print("\nStarting metrics evaluation...")
        
        # Compute FID
        print("\nComputing FID score...")
        fid = self.compute_fid(generated_path)
        
        # Compute Precision-Recall
        print("\nComputing Precision-Recall metrics...")
        precision, recall = self.compute_precision_recall(generated_path)
        
        results = {
            'fid_score': fid,
            'precision': precision,
            'recall': recall,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return results

def save_evaluation_results(results: Dict[str, float], output_path: str):
    """Save evaluation results to a JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {output_path}")

def print_evaluation_results(results: Dict[str, float]):
    """Print evaluation results in a formatted way"""
    print("\nEvaluation Results:")
    print(f"FID Score: {results['fid_score']:.2f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated samples using multiple metrics')
    parser.add_argument('--real_data_path', type=str, default='data/MNIST',
                      help='Path to real MNIST data')
    parser.add_argument('--samples_dir', type=str, required=True,
                      help='Directory containing generated samples')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for evaluation')
    parser.add_argument('--output_path', type=str, default='results/metrics.json',
                      help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MetricsEvaluator(
        real_data_path=args.real_data_path,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    results = evaluator.evaluate_samples(args.samples_dir)
    
    # Save and print results
    save_evaluation_results(results, args.output_path)
    print_evaluation_results(results)