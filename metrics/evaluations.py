# metrics/evaluations.py

import torch
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_precision_recall(real_features, fake_features, k=1):
    real_features = torch.from_numpy(real_features)
    fake_features = torch.from_numpy(fake_features)

    real_nearest_neighbors = torch.cdist(real_features, fake_features).topk(k, largest=False).indices
    fake_nearest_neighbors = torch.cdist(fake_features, real_features).topk(k, largest=False).indices

    precision = (real_nearest_neighbors == fake_nearest_neighbors.T).float().mean().item()
    recall = (fake_nearest_neighbors == real_nearest_neighbors.T).float().mean().item()

    return precision, recall
