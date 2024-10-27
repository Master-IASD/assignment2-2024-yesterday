# metrics/evaluate.py

import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from evaluations import calculate_fid, calculate_precision_recall

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import Generator
from utils import load_model

def extract_real_features(model, dataloader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.cuda()
            features.append(model(images).cpu().numpy())
    return np.vstack(features)

def extract_fake_features(model, num_samples=10000, batch_size=64):
    model.eval()
    fake_features = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, 100).cuda()  # Generate random noise vector
            generated_images = model(z)  # Forward pass through the generator
            fake_features.append(generated_images.cpu().numpy())  # Collect generated features
    return np.vstack(fake_features)

def main():
    # Load the pre-trained generator model
    model = Generator(g_output_dim=784).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Extract features
    real_features = extract_real_features(model, dataloader)  # Using real MNIST images
    fake_features = extract_fake_features(model)  # Using generated images

    # Calculate FID, precision, and recall
    fid = calculate_fid(real_features, fake_features)
    precision, recall = calculate_precision_recall(real_features, fake_features)

    print(f'FID: {fid}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

if __name__ == '__main__':
    main()
