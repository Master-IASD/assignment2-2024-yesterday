import torch 
import torchvision
import os
import argparse
import torch.optim as optim
from torch.distributions import Normal, Independent
import numpy as np
from pyro.infer.mcmc import MCMC, NUTS
import pyro
from model import Generator,Discriminator
from utils import load_model

def load_model_D(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D
def truncated_normal(shape, mean=0.0, std=1.0, truncation=2.0):

    z = torch.randn(shape).cuda() * std + mean
    
    while True:
        mask = (z < -truncation) | (z > truncation)
        if not mask.any():
            break
        z[mask] = torch.randn(mask.sum()).cuda() * std + mean
    
    return z
def DDLS_sampling(generator, discriminator, z_dim, eps, num_iter, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    cur_z_arr = []
    for i in range(batch_size):
        loc = torch.zeros(z_dim, device=device)
        scale = torch.ones(z_dim, device=device)
        normal = Normal(loc, scale)
        diagn = Independent(normal, 1)
        cur_z = diagn.sample()
        cur_z_arr.append(cur_z.clone())
    cur_z_arr = torch.stack(cur_z_arr, dim=0).to(device)
    cur_z_arr.requires_grad_(True)  

    for i in range(num_iter):

        generated_samples = generator(cur_z_arr)
        gan_part = -discriminator(generated_samples).squeeze() 
        latent_part = -Normal(torch.zeros(z_dim, device=device), torch.ones(z_dim, device=device)).log_prob(cur_z_arr).sum(dim=1)

        energy = gan_part + latent_part
        energy = energy.sum()  
        energy.backward()  

        # Langevin 
        with torch.no_grad():
            noise = torch.randn_like(cur_z_arr).to(device)
            cur_z_arr -= (eps / 2) * cur_z_arr.grad - (eps ** 0.5) * noise

        cur_z_arr.grad.zero_()

    return cur_z_arr


def calculate_energy(z):

    z = z['points']
    return 0.5 * torch.sum(z ** 2, dim=-1)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    # model = Generator(g_output_dim = mnist_dim).cuda()
    # model=Generator(image_size=784,
    #              hidden_dim=400,
    #              z_dim=20).cuda()
    # model = load_model(model, 'checkpoints_gan')
    # model = torch.nn.DataParallel(model).cuda()
    # model.eval()

    # print('Model loaded.')
    generator = Generator(g_output_dim = mnist_dim).cuda()
    discriminator=Discriminator(mnist_dim).cuda()
    generator = load_model(generator, 'checkpoints')
    discriminator=load_model_D(discriminator, 'checkpoints')

    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)
    eps = 0.1 
    num_iter = 100  # Langevin 
    batch_size = 2048  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 0
    kernel = NUTS(potential_fn=calculate_energy)
    mcmc = MCMC(kernel=kernel, num_samples=5000, initial_params={'points': torch.zeros(100).to(device)}, num_chains=1)

    mcmc.run()
    sampled_z = mcmc.get_samples()['points']  
    with torch.no_grad():
        # while n_samples<10000:
        for i in range(0, len(sampled_z), batch_size):
            # z = torch.randn(args.batch_size, 100).cuda()
            # # z =truncated_normal((args.batch_size, 20), truncation=2.0).cuda()
            # x = model(z)
            # x = x.reshape(args.batch_size, 28, 28)
            # sampled_z = DDLS_sampling(generator, discriminator, 100, eps, num_iter, batch_size)
            # generated_images = generator(sampled_z)
            # generated_images = generated_images.reshape(batch_size, 1, 28, 28)
            z_batch = sampled_z[i:i+batch_size].to(device)

            generated_images = generator(z_batch)
            generated_images = generated_images.reshape(generated_images.size(0), 1, 28, 28) 

       
            for k in range(generated_images.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(generated_images[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1
            if n_samples >= 10000:
                break  


    
