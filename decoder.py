import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    def __init__(self, c_last, img_channels=3):
        super(Decoder, self).__init__()

        # Decoder mirrors the encoder, Tconvs with stride 2 for upsampling
        self.tconv1 = nn.ConvTranspose2d(c_last, 64, kernel_size=5, stride=1, padding=2)
        self.prelu1 = nn.PReLU(64)

        self.tconv2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.prelu2 = nn.PReLU(64)

        self.tconv3 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.prelu3 = nn.PReLU(64)

        # 32x32 -> 64x64
        self.tconv4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.prelu4 = nn.PReLU(32)

        # 64x64 -> 128x128
        self.tconv5 = nn.ConvTranspose2d(32, img_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z_hat, bottleneck_shape):
        # 1. Reshape the flat vector z_hat back into its 3D feature map
        x_hat = z_hat.view(z_hat.size(0), *bottleneck_shape)

        # 2. Pass through all transpose conv layers
        x_hat = self.prelu1(self.tconv1(x_hat))
        x_hat = self.prelu2(self.tconv2(x_hat))
        x_hat = self.prelu3(self.tconv3(x_hat))
        x_hat = self.prelu4(self.tconv4(x_hat))

        # 3. Final layer + sigmoid activation
        x_hat = self.tconv5(x_hat)
        x_hat = torch.sigmoid(x_hat) # Output [0, 1]

        return x_hat

def receive_all(sock, n):
    """Helper function to ensure all n bytes are received."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            # Connection closed prematurely
            return None
        data.extend(packet)
    return data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C_LAST = 8  
MODEL_PATH = 'decoder_weights_snr_19db.pth' 
HOST='0.0.0.0'
PORT=6000
bottleneck_shape = (C_LAST, 256, 256)
decoder_model = Decoder(c_last=C_LAST)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
decoder_model.load_state_dict(state_dict)
decoder_model.to(DEVICE)
decoder_model.eval()
print(f"decoder model loaded successfully from {MODEL_PATH} and set to evaluation mode.")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"\n🚀 Listening for connection on port {PORT}...")
    
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}. Ready to receive Z vector.")

        # 1. Receive Header (4 bytes for data length)
        header_bytes = receive_all(conn, 4)
        if not header_bytes:
            print("Connection closed by sender.")
            exit()
            
        data_len = struct.unpack('!I', header_bytes)[0]
        print(f"Expecting {data_len} bytes of Z vector data...")

        # 2. Receive the Z vector data
        z_bytes = receive_all(conn, data_len)
        if not z_bytes:
            print("Failed to receive full data.")
            exit()
        
        # 3. Deserialize back to NumPy array (float32)
        z_numpy = np.frombuffer(z_bytes, dtype=np.float32)
        
        # 4. Convert to PyTorch tensor (shape: [524288])
        z_tensor = torch.from_numpy(z_numpy).unsqueeze(0).to(DEVICE)
        
        print(f"✅ Data received successfully. Z vector shape: {z_tensor.shape}")

with torch.no_grad():
    x_hat_norm = decoder_model(z_tensor, bottleneck_shape)
    x_hat = x_hat_norm.squeeze(0).cpu().permute(1, 2, 0)
    reconstructed_img_np = (x_hat.numpy() * 255.0).astype(np.uint8)


reconstructed_pil = Image.fromarray(reconstructed_img_np)

plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_pil)
plt.title(f"Reconstructed Image")
plt.axis('off')
plt.show()
print("\nReconstruction complete and visualized.")