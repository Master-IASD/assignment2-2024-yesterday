import torch
import os

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output = D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))

def load_model(model, folder, model_type):
    ckpt = torch.load(os.path.join(folder, f'{model_type}.pth'))
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return model

# WGAN utils:

def WGAN_D_train(x, G, D, D_optimizer, gp_weight=10):
    """Wasserstein GAN discriminator training step with gradient penalty"""
    D.zero_grad()
    batch_size = x.size(0)
    
    # Train with real
    x_real = x.cuda()
    D_real = D(x_real)
    
    # Train with fake
    z = torch.randn(batch_size, 100).cuda()
    x_fake = G(z)
    D_fake = D(x_fake)
    
    # Gradient penalty
    alpha = torch.rand(batch_size, 1).cuda()
    x_interpolated = (alpha * x_real + (1 - alpha) * x_fake.detach()).requires_grad_(True)
    D_interpolated = D(x_interpolated)
    
    gradients = torch.autograd.grad(
        outputs=D_interpolated,
        inputs=x_interpolated,
        grad_outputs=torch.ones_like(D_interpolated).cuda(),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    # Wasserstein loss
    D_loss = -torch.mean(D_real) + torch.mean(D_fake) + gp_weight * gradient_penalty
    
    D_loss.backward()
    D_optimizer.step()
    
    return D_loss.item()

def WGAN_G_train(x, G, D, G_optimizer):
    """Wasserstein GAN generator training step"""
    G.zero_grad()
    
    z = torch.randn(x.shape[0], 100).cuda()
    fake_imgs = G(z)
    G_loss = -torch.mean(D(fake_imgs))
    
    G_loss.backward()
    G_optimizer.step()
    
    return G_loss.item()

def rejection_sampling(G, D, n_samples, threshold=0.7, batch_size=100):
    """Perform rejection sampling using discriminator scores"""
    accepted_samples = []
    accepted_scores = []
    
    with torch.no_grad():
        while len(accepted_samples) < n_samples:
            # Generate a batch
            z = torch.randn(batch_size, 100).cuda()
            fake_samples = G(z)
            d_scores = D(fake_samples)
            
            # Normalize scores to [0,1] range for rejection sampling
            d_scores = torch.sigmoid(d_scores)
            
            # Accept samples based on threshold
            accepted_mask = d_scores.squeeze() > threshold
            
            if accepted_mask.any():
                accepted_samples.append(fake_samples[accepted_mask])
                accepted_scores.append(d_scores[accepted_mask])
                
    # Concatenate all accepted samples
    accepted_samples = torch.cat(accepted_samples, dim=0)[:n_samples]
    accepted_scores = torch.cat(accepted_scores, dim=0)[:n_samples]
    
    return accepted_samples, accepted_scores

def save_wgan_models(G, D, folder):
    """Save WGAN models with distinct names"""
    torch.save(G.state_dict(), os.path.join(folder, 'G_wgan.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D_wgan.pth'))

