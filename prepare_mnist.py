import os
import uuid
import torchvision
from torchvision import datasets, transforms

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def prepare_mnist_dataset():
    # Download and prepare the MNIST dataset
    train_dataset = datasets.MNIST(root='data/MNIST', train=True, transform=transform, download=True)

    # Create directories for each class
    os.makedirs('data/MNIST/train', exist_ok=True)
    for i in range(10):
        os.makedirs(f'data/MNIST/train/{i}', exist_ok=True)

    # Save images to the corresponding directories
    for img, label in train_dataset:
        img = transforms.ToPILImage()(img)
        img.save(f'data/MNIST/train/{label}/{uuid.uuid4()}.png')

if __name__ == '__main__':
    prepare_mnist_dataset()
