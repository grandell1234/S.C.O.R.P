
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, num_classes=10):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), -1)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes=10):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size + num_classes, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Function to generate an image based on the specified number
def generate_image(generator, specified_number, latent_dim, device):
    generator.eval()
    z = torch.randn(1, latent_dim).to(device)
    label = torch.tensor([specified_number], dtype=torch.long).to(device)
    one_hot_label = nn.functional.one_hot(label, num_classes=10).float()

    # Concatenate noise vector and one-hot label
    z_with_label = torch.cat([z, one_hot_label], dim=1)

    # Generate the image
    gen_img = generator(z_with_label)
    gen_img = gen_img.view(1, 1, 28, 28)  # Adjust the view here

    return gen_img

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    latent_dim = 100
    img_size = 28 * 28

    # Initialize the Generator and Discriminator
    generator = Generator(latent_dim, img_size)
    discriminator = Discriminator(img_size)

    # Load the trained models
    generator.load_state_dict(torch.load("mnist_generator_epoch_999.pt", map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load("mnist_discriminator_epoch_999.pt", map_location=torch.device('cpu')))

    # Specify the number for image generation (0-9)
    specified_number = 1

    # Generate and save the image in the 'images' directory
    generated_image = generate_image(generator, specified_number, latent_dim, device='cpu')

    # Create the 'images' directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Save the image in the 'images' directory
    save_image(generated_image, os.path.join("images", f"generated_image_{specified_number}.png"), normalize=True)
