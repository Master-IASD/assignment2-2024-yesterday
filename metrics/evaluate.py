# metrics/evaluate.py

import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from evaluations import calculate_fid, calculate_precision_recall
from torchvision import models

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
            images = images.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel (pseudo-RGB)
            images = torch.nn.functional.interpolate(images, size=(299, 299))  # Resize for InceptionV3
            features.append(model(images).cpu().numpy())
    return np.vstack(features)

def extract_fake_features(generator, num_samples=10000, batch_size=64):
    generator.eval()
    fake_features = []
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            z = torch.randn(batch_size, 100).cuda()  # Generate random noise vector
            generated_images = generator(z)  # Generate fake images
            generated_images = generated_images.view(batch_size, 1, 28, 28)  # Reshape for feature extraction
            generated_images = generated_images.repeat(1, 3, 1, 1)  # Convert to 3-channel (pseudo-RGB)
            generated_images = torch.nn.functional.interpolate(generated_images, size=(299, 299))  # Resize
            fake_features.append(generated_images.cpu().numpy())
    return np.vstack(fake_features)

def main():
    # Load the pre-trained generator model
    generator = Generator(g_output_dim=784).cuda()
    generator = load_model(generator, 'checkpoints')
    generator = torch.nn.DataParallel(generator).cuda()
    generator.eval()

    # Use a pretrained InceptionV3 model for feature extraction
    feature_extractor = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1).cuda()
    feature_extractor.fc = torch.nn.Identity()  # Remove final classification layer
    feature_extractor = torch.nn.DataParallel(feature_extractor).cuda()

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    dataset = MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Extract features from real and generated images
    real_features = extract_real_features(feature_extractor, dataloader)  # Using InceptionV3 for MNIST images
    fake_features = extract_fake_features(generator)  # Using generated images

    # Calculate FID, precision, and recall
    fid = calculate_fid(real_features, fake_features)
    precision, recall = calculate_precision_recall(real_features, fake_features)

    print(f'FID: {fid}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

if __name__ == '__main__':
    main()
