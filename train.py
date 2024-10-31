import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision


from model import Generator, Discriminator, WDiscriminator 
from utils import D_train, G_train, save_models, WGAN_D_train, WGAN_G_train, save_wgan_models, rejection_sampling

def train_wgan(args):
    """Training function for WGAN with rejection sampling"""
    # Data loading code remains the same as in main()
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Data Pipeline
    print("Dataset loading...")
    # MNIST Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
    )

    train_dataset = datasets.MNIST(
        root="data/MNIST/", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="data/MNIST/", train=False, transform=transform, download=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    print("Dataset Loaded.")

    print("Model Loading...")

    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).cuda()  # Use the same generator architecture
    D = WDiscriminator(mnist_dim).cuda()  # Use the new Wasserstein discriminator
    
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    
    # WGAN typically uses RMSprop or Adam with specific parameters
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    for epoch in trange(1, args.epochs + 1, leave=True):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            
            # Train discriminator more frequently (typically 5 times per G update)
            for _ in range(5):
                WGAN_D_train(x, G, D, D_optimizer)
            
            WGAN_G_train(x, G, D, G_optimizer)
        
        if epoch % 10 == 0:
            save_wgan_models(G, D, 'checkpoints')
            
            # Generate samples with rejection sampling
            samples, scores = rejection_sampling(G, D, n_samples=100)
            # Save some samples for visualization
            for i, sample in enumerate(samples[:10]):
                torchvision.utils.save_image(
                    sample.view(28, 28),
                    f'training_samples/wgan_epoch_{epoch}_sample_{i}.png'
                )
    
    return G, D

# Add these lines at the end of the main block in train.py:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Normalizing Flow.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of mini-batches for SGD"
    )
    #New Arg to differentiate between training VanillanGAN and WGAN
    parser.add_argument("--model", type=str, default="vanilla",
                      choices=["vanilla", "wgan"],
                      help="Type of GAN to train (vanilla or wgan)")
    
    args = parser.parse_args()
    
    if args.model == "vanilla":
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        #For testing purposes
        os.makedirs("training_samples", exist_ok=True)  # Create a directory for training samples

        # Data Pipeline
        print("Dataset loading...")
        # MNIST Dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
        )

        train_dataset = datasets.MNIST(
            root="data/MNIST/", train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(
            root="data/MNIST/", train=False, transform=transform, download=False
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=args.batch_size, shuffle=False
        )
        print("Dataset Loaded.")

        print("Model Loading...")
        mnist_dim = 784
        G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
        D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()

        # model = DataParallel(model).cuda()
        print("Model loaded.")
        # Optimizer

        # define loss
        criterion = nn.BCELoss()

        # define optimizers
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

        print("Training Vanilla GAN...")

        n_epoch = args.epochs
        for epoch in trange(1, n_epoch + 1, leave=True):
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.view(-1, mnist_dim)
                D_train(x, G, D, D_optimizer, criterion)
                G_train(x, G, D, G_optimizer, criterion)

            if epoch % 10 == 0:
                save_models(G, D, "checkpoints")

        print("Training done")

    else:
        print("Training WGAN with Rejection Sampling...")
        train_wgan(args)

