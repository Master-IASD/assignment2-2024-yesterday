import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from scipy import linalg
from tqdm import tqdm

class InceptionV3Features(nn.Module):
    def __init__(self):
        super(InceptionV3Features, self).__init__()
        # Load pretrained InceptionV3 model
        inception = models.inception_v3(pretrained=True)
        # We only need features up to the last average pooling layer
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2
        )
        
    def forward(self, x):
        # Resize input to inception expected size
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert grayscale to RGB by repeating channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features

def calculate_activation_statistics(dataloader, model):
    """Calculate mean and covariance of features."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating activation statistics"):
            # Handle both tuple/list from DataLoader and direct tensors
            if isinstance(batch, (tuple, list)):
                batch = batch[0]  # Get the data from the batch tuple
            batch = batch.cuda()
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma

def calculate_fid(real_loader, generated_loader, batch_size=50):
    """Calculate Fréchet Inception Distance between real and generated images."""
    model = InceptionV3Features().cuda()
    
    # Calculate statistics for real images
    mu1, sigma1 = calculate_activation_statistics(real_loader, model)
    
    # Calculate statistics for generated images
    mu2, sigma2 = calculate_activation_statistics(generated_loader, model)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return float(fid)

def calculate_precision_recall(real_features, generated_features, k=3, threshold=5e-3):
    """Calculate precision and recall metrics."""
    # Normalize features
    real_features = F.normalize(real_features, dim=1)
    generated_features = F.normalize(generated_features, dim=1)
    
    # Calculate pairwise distances
    real_distances = torch.cdist(real_features, real_features)
    gen_distances = torch.cdist(generated_features, real_features)
    
    # Find k-nearest neighbors
    real_nearest = torch.topk(real_distances, k=k+1, dim=1, largest=False)[0][:, 1:]
    gen_nearest = torch.topk(gen_distances, k=k, dim=1, largest=False)[0]
    
    # Calculate precision and recall
    real_radius = torch.max(real_nearest, dim=1)[0]
    precision = torch.mean((gen_nearest <= real_radius.unsqueeze(1)).float())
    recall = torch.mean((gen_distances.min(dim=0)[0] <= real_radius).float())
    
    return precision.item(), recall.item()