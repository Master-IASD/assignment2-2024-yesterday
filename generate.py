import torch
import torchvision
import os
import argparse

from model import Generator, Discriminator
from utils import load_model

'''
def truncated_normal(shape, mean=0.0, std=1.0, truncation=2.0):

    z = torch.randn(shape).cuda() * std + mean
    
    while True:
        mask = (z < -truncation) | (z > truncation)
        if not mask.any():
            break
        z[mask] = torch.randn(mask.sum()).cuda() * std + mean   
    return z
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Generator model and load weights
    G = Generator(g_output_dim=mnist_dim).to(device)
    G = load_model(G, 'checkpoints', 'G_W-GAN-GP')
    G = torch.nn.DataParallel(G).to(device)
    G.eval()

    print('Model loaded.')
    print('Start Generating')

    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            z = torch.randn(args.batch_size, 100, device=device)  # Ensure latent vector uses the correct device
            x = G(z).reshape(args.batch_size, 1, 28, 28)  # Reshape for image saving

            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(x[k], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1
                else:
                    break

    print('Sample generation complete.')