import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image

class Encoder(nn.Module):
    def __init__(self, c_last):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.prelu1 = nn.PReLU(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.prelu2 = nn.PReLU(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.prelu4 = nn.PReLU(64)
        self.conv5 = nn.Conv2d(64, c_last, kernel_size=5, stride=1, padding=2)
        self.prelu5 = nn.PReLU(c_last)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x_out = self.prelu5(self.conv5(x))
        batch_size = x_out.size(0)
        z_tilde = x_out.view(batch_size, -1)
        num_elements = z_tilde.size(1)
        k = num_elements / 2
        z_norm = torch.norm(z_tilde, p=2, dim=1, keepdim=True)
        sqrt_k = torch.sqrt(torch.tensor(k, dtype=torch.float32).to(x.device))
        z = sqrt_k * (z_tilde / (z_norm + 1e-8))
        return z, x_out.shape[1:]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C_LAST = 8  
MODEL_PATH = 'encoder_weights_snr_19db.pth' 
encoder_model = Encoder(c_last=C_LAST)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
encoder_model.load_state_dict(state_dict)
encoder_model.to(DEVICE)
encoder_model.eval()
print(f"Encoder model loaded successfully from {MODEL_PATH} and set to evaluation mode.")
IMAGE_SIZE = 1024
image_path = "test.jpg"
pil_image = Image.open(image_path).convert('RGB')

preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

x_norm = preprocess(pil_image).unsqueeze(0).to(DEVICE)
original_img_norm = x_norm.squeeze(0).cpu().permute(1, 2, 0).numpy()

with torch.no_grad():
    z, bottleneck_shape = encoder_model(x_norm)

z_flat=z.squeeze(0)

inphase=z_flat[0::2]
quadrature=z_flat[1::2]

print(inphase,quadrature)



