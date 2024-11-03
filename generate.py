import torch
import torchvision
import os
import argparse

from model import Generator, Discriminator, WDiscriminator
from utils import load_model, rejection_sampling

def normalize_discriminator_scores(scores, model_type):
    """Normalize discriminator scores based on the model type"""
    if model_type == "vanilla":
        # Vanilla GAN already outputs probabilities (0-1)
        return scores
    elif model_type == "wgan":
        # Convert WGAN scores to pseudo-probabilities using sigmoid
        return torch.sigmoid(scores)
    else:
        # For future GAN implementations, add normalization logic here
        return scores

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
    parser = argparse.ArgumentParser(description='Generate Samples from GAN Models.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--model", type=str, default="wgan",
                      choices=["vanilla", "wgan"],  # Add new models here
                      help="Which model to use for generation")
    parser.add_argument("--rejection_sampling", action="store_true", default=True,
                      help="Whether to use rejection sampling")
    parser.add_argument("--threshold", type=float, default=0.7,
                      help="Threshold for rejection sampling (0-1)")
    parser.add_argument("--output_dir", type=str, default="samples",
                      help="Directory to save generated samples")
    args = parser.parse_args()

    print('Model Loading...')
    mnist_dim = 784

    # Load Generator 
    G = Generator(g_output_dim=mnist_dim).cuda()
    
    # Model-specific loading
    model_configs = {
        "vanilla": {
            "g_path": "G",
            "d_path": "D",
            "discriminator": Discriminator(mnist_dim)
        },
        "wgan": {
            "g_path": "G_wgan",
            "d_path": "D_wgan",
            "discriminator": WDiscriminator(mnist_dim)
        }
    }

    config = model_configs[args.model]
    G = load_model(G, 'checkpoints', config["g_path"])
    G = torch.nn.DataParallel(G).cuda()
    G.eval()

    D = None
    if args.rejection_sampling:
        D = config["discriminator"].cuda()
        D = load_model(D, 'checkpoints', config["d_path"])
        D = torch.nn.DataParallel(D).cuda()
        D.eval()

    print(f'Model loaded. Using {args.model} GAN model')
    print(f'Rejection sampling: {"enabled" if args.rejection_sampling else "disabled"}')


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

    # sample_dir = os.path.join(args.output_dir, f'{args.model}_gan')
    # if args.rejection_sampling:
    #     sample_dir += f'_rejection_{args.threshold}'
    # os.makedirs(sample_dir, exist_ok=True)

    print('Start Generating')

    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        if args.rejection_sampling and D is not None:
            print(f'Using rejection sampling with threshold {args.threshold}...')
            accepted_samples = []
            
            while len(accepted_samples) < 10000:
                # Generate batch
                z = torch.randn(args.batch_size, 100).cuda()
                fake_samples = G(z)
                
                # Get discriminator scores
                d_scores = D(fake_samples)
                # Normalize scores based on model type
                d_scores = normalize_discriminator_scores(d_scores, args.model)
                
                # Apply rejection sampling
                accepted_mask = d_scores.squeeze() > args.threshold
                
                if accepted_mask.any():
                    accepted_batch = fake_samples[accepted_mask]
                    accepted_samples.append(accepted_batch)
                    
                    # Save accepted samples
                    for idx, sample in enumerate(accepted_batch):
                        if len(accepted_samples) * args.batch_size + idx < 10000:
                            torchvision.utils.save_image(
                                sample.view(28, 28),
                                os.path.join('samples', f'{n_samples}.png')
                            )
                            n_samples += 1
                        else:
                            break
                
                if n_samples >= 10000:
                    break
                
                if n_samples % 100 == 0:
                    print(f'Generated {n_samples}/10000 samples')
        else:
            # Standard generation without rejection sampling
            while n_samples < 10000:
                z = torch.randn(args.batch_size, 100).cuda()
                x = G(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples < 10000:
                        torchvision.utils.save_image(
                            x[k:k+1],
                            os.path.join('samples', f'{n_samples}.png')
                        )
                        n_samples += 1

    print('Sample generation complete.')