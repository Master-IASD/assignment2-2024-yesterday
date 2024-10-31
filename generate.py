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

    G = Generator(g_output_dim=mnist_dim).cuda()
    G = load_model(G, 'checkpoints', 'G')
    G = torch.nn.DataParallel(G).cuda()
    G.eval()

    D = Discriminator(d_input_dim=mnist_dim).cuda()
    D = load_model(D, 'checkpoints', 'D')
    D = torch.nn.DataParallel(D).cuda()
    D.eval()

    # model = Generator(g_output_dim = mnist_dim).cuda()
    # model = load_model(model, 'checkpoints')
    # model = torch.nn.DataParallel(model).cuda()
    # model.eval()
    
    # Load the pretrained Generator model
    # model_path = 'checkpoints/G_Vanilla.pth'
    # model = torch.load(model_path)  # Load model directly
    # model = model.to('cuda')  # Place model on the primary GPU
    # model = torch.nn.DataParallel(model)  # Wrap model with DataParallel for multi-GPU support
    # model.eval()  # Set model to evaluation mode

    print('Model loaded.')
    
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            z = torch.randn(args.batch_size, 100).cuda()
            # z =truncated_normal((args.batch_size, 100), truncation=2.0).cuda()
            x = G(z) # debug model to G
            x = x.reshape(args.batch_size, 28, 28)

            # Rejection Sampling
            D_output = D(x.view(args.batch_size, -1))
            acceptance_prob = torch.sigmoid(D_output).squeeze()

            for k in range(x.shape[0]):
                if n_samples < 10000:
                    if acceptance_prob[k] > 0.5:  # Accept if discriminator output is above a threshold
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))
                        n_samples += 1

    print('Sample generation complete.')
