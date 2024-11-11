import torch
import torchvision
import os
import argparse
from model import Generator, Discriminator, WDiscriminator
from utils import load_model, rejection_sampling, generate_with_obrs

def normalize_discriminator_scores(scores, model_type):
    """Normalize discriminator scores based on the model type"""
    if model_type == "vanilla":
        return scores
    elif model_type == "wgan":
        return torch.sigmoid(scores)
    else:
        return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Samples from GAN Models.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for generation.")
    parser.add_argument("--model", type=str, default="vanilla",
                      choices=["vanilla", "wgan"],
                      help="Which model to use for generation")
    parser.add_argument("--rejection_sampling", action="store_true", default=True,
                      help="Whether to use rejection sampling")
    parser.add_argument("--sampling_method", type=str, default="obrs",
                      choices=["standard", "rejection", "obrs"],
                      help="Sampling method to use")
    parser.add_argument("--budget_K", type=float, default=6.0,
                      help="Budget parameter K for OBRS")
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

    # Load Discriminator if needed for rejection sampling or OBRS
    D = None
    if args.rejection_sampling or args.sampling_method == "obrs":
        D = config["discriminator"].cuda()
        D = load_model(D, 'checkpoints', config["d_path"])
        D = torch.nn.DataParallel(D).cuda()
        D.eval()

    print(f'Model loaded. Using {args.model} GAN model')
    print(f'Sampling method: {args.sampling_method}')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        if args.sampling_method == "obrs":
            samples = generate_with_obrs(G, D, 
                                      n_samples=10000,
                                      budget_K=args.budget_K,
                                      batch_size=args.batch_size,
                                      output_dir=args.output_dir)
        elif args.rejection_sampling and D is not None:
            print(f'Using rejection sampling with threshold {args.threshold}...')
            accepted_samples = []
            
            while len(accepted_samples) < 10000:
                z = torch.randn(args.batch_size, 100).cuda()
                fake_samples = G(z)
                d_scores = D(fake_samples)
                d_scores = normalize_discriminator_scores(d_scores, args.model)
                accepted_mask = d_scores.squeeze() > args.threshold
                
                if accepted_mask.any():
                    accepted_batch = fake_samples[accepted_mask]
                    accepted_samples.append(accepted_batch)
                    
                    for idx, sample in enumerate(accepted_batch):
                        if n_samples < 10000:
                            torchvision.utils.save_image(
                                sample.view(28, 28),
                                os.path.join('samples', f'{n_samples}.png')
                            )
                            n_samples += 1
                
                if n_samples >= 10000:
                    break
                
                if n_samples % 100 == 0:
                    print(f'Generated {n_samples}/10000 samples')
        else:
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