import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(110, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))
# class Generator(nn.Module):
#     """ Generator. Input is noise, output is a generated image.
#     """
#     def __init__(self, image_size, hidden_dim, z_dim):
#         super().__init__()
#         self.linear = nn.Linear(z_dim, hidden_dim)
#         self.generate = nn.Linear(hidden_dim, image_size)

#     def forward(self, x):
#         activated = F.relu(self.linear(x))
#         generation = torch.sigmoid(self.generate(activated))
#         return generation


# class Discriminator(nn.Module):
#     """ Discriminator. Input is an image (real or generated),
#     output is P(generated).
#     """
#     def __init__(self, image_size, hidden_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Linear(image_size, hidden_dim)
#         self.discriminate = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         activated = F.relu(self.linear(x))
#         discrimination = torch.sigmoid(self.discriminate(activated))
#         return discrimination

