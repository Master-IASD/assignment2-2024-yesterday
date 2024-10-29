# evaluations.py
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
        # Use the new weights parameter instead of pretrained
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        inception = models.inception_v3(weights=weights)
        # We only need features up to the last average pooling layer
        self.feature_extractor = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2
        )
        
    @torch.no_grad()
    def forward(self, x):
        # Resize input to inception expected size
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # Convert grayscale to RGB by repeating channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # Extract features
        features = self.feature_extractor(x)
        return features.view(features.size(0), -1)

def calculate_activation_statistics(dataloader, model, device):
    """Calculate mean and covariance of features."""
    features_list = []
    model = model.to(device)
    model.eval()
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating activation statistics"):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(device)
                feat = model(batch).cpu().numpy()
                features_list.append(feat)
                
                # Clear cache periodically
                if len(features_list) % 10 == 0:
                    torch.cuda.empty_cache()
        
        features = np.concatenate(features_list, axis=0)
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    except Exception as e:
        print(f"Error in calculate_activation_statistics: {str(e)}")
        raise

def calculate_fid(real_loader, generated_loader, batch_size=50, device='cuda'):
    """Calculate Fréchet Inception Distance between real and generated images."""
    try:
        model = InceptionV3Features().to(device)
        print("Calculating statistics for real images...")
        mu1, sigma1 = calculate_activation_statistics(real_loader, model, device)
        
        print("Calculating statistics for generated images...")
        mu2, sigma2 = calculate_activation_statistics(generated_loader, model, device)
        
        print("Computing FID score...")
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)
    
    except Exception as e:
        print(f"Error in calculate_fid: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

@torch.no_grad()
def calculate_precision_recall(real_features, generated_features, k=3, threshold=5e-3):
    """Calculate precision and recall metrics."""
    try:
        # Move to CPU for memory efficiency
        real_features = real_features.cpu()
        generated_features = generated_features.cpu()
        
        # Normalize features in smaller batches
        batch_size = 1000
        for i in range(0, len(real_features), batch_size):
            batch = real_features[i:i+batch_size]
            real_features[i:i+batch_size] = F.normalize(batch, dim=1)
        
        for i in range(0, len(generated_features), batch_size):
            batch = generated_features[i:i+batch_size]
            generated_features[i:i+batch_size] = F.normalize(batch, dim=1)
        
        print("Computing pairwise distances...")
        # Calculate distances in batches
        real_distances = []
        for i in tqdm(range(0, len(real_features), batch_size)):
            batch = real_features[i:i+batch_size]
            dist = torch.cdist(batch, real_features)
            real_distances.append(dist)
        real_distances = torch.cat(real_distances, dim=0)
        
        gen_distances = []
        for i in tqdm(range(0, len(generated_features), batch_size)):
            batch = generated_features[i:i+batch_size]
            dist = torch.cdist(batch, real_features)
            gen_distances.append(dist)
        gen_distances = torch.cat(gen_distances, dim=0)
        
        print("Computing nearest neighbors...")
        real_nearest = torch.topk(real_distances, k=k+1, dim=1, largest=False)[0][:, 1:]
        gen_nearest = torch.topk(gen_distances, k=k, dim=1, largest=False)[0]
        
        real_radius = torch.max(real_nearest, dim=1)[0]
        precision = torch.mean((gen_nearest <= real_radius.unsqueeze(1)).float())
        recall = torch.mean((gen_distances.min(dim=0)[0] <= real_radius).float())
        
        return precision.item(), recall.item()
    
    except Exception as e:
        print(f"Error in calculate_precision_recall: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()