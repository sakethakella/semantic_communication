# Semantic Communication

This repository contains a minimal example of a semantic communication pipeline implemented in PyTorch. The system encodes an input image into a compact bottleneck vector (the semantic representation), sends that vector over a TCP socket to a remote host, and decodes it back to an image.

**Project Goals:**

- Demonstrate a simple encoder/decoder (autoencoder-like) pipeline.
- Show how to serialize/transfer the latent (bottleneck) vector over a network socket.
- Provide an easy-to-run example to experiment with semantic communication ideas.

**Contents**

- `encoder.py`: Encodes an image into a normalized bottleneck vector and sends it (I/Q interleaved) via TCP.
- `decoder.py`: Receives the serialized vector on a listening TCP socket, reconstructs the image using the decoder model, and visualizes it.
- `encoder_weights_snr_19db.pth` and `decoder_weights_snr_19db.pth`: Trained model weights expected by the scripts.
- `requirements.txt`: Python package requirements.

**How it works (high level)**

- `encoder.py` reads `test.jpg`, resizes to 1024×1024, runs the `Encoder` model to produce a bottleneck feature map.
- The bottleneck is flattened and L2-normalized to produce a vector `z` (with interleaved in-phase and quadrature components when sending).
- `encoder.py` connects to `decoder.py` over TCP, sends a 4-byte big-endian unsigned int header describing the payload length, then sends the raw float32 bytes of the interleaved vector.
- `decoder.py` accepts a connection, reads the 4-byte header, receives the payload fully, converts the bytes to float32, constructs a tensor, reshapes to the expected bottleneck shape, and runs the `Decoder` to reconstruct an RGB image. The reconstructed image is displayed with `matplotlib`.

**Important implementation details**

- Image size: The encoder expects/sends images resized to `1024x1024`.
- Bottleneck channels: `C_LAST = 8` by default. In the code the expected `bottleneck_shape` used by the `Decoder` is `(8, 256, 256)` (i.e. 8 channels, 256×256 spatial) which matches 1024/4 = 256 if the encoder downscales by factor 4.
- Serialization format: float32 array, interleaved I/Q: `z_numpy[0::2] = inphase`, `z_numpy[1::2] = quadrature`.
- Network framing: 4-byte header `struct.pack('!I', len(z_bytes))` followed immediately by `z_bytes` payload.

**Requirements**

- Python 3.8+ (tested with 3.8–3.11)
- `torch`, `torchvision`, `numpy`, `Pillow`, `matplotlib`
- See `requirements.txt` for pinned versions.

**Run instructions (example)**

1. Prepare environment and install dependencies (PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the decoder on the receiving machine (or locally in a separate terminal). By default it listens on `0.0.0.0:6000`:

```powershell
python decoder.py
```

3. Run the encoder on the sending machine and point `HOST` in `encoder.py` to the receiver IP (or `localhost` for local runs):

```powershell
python encoder.py
```

The encoder will open `test.jpg`, produce the latent vector, and send the data. The decoder will reconstruct the image and display it.

**Files and key symbols**

- `encoder.py`:
  - Class `Encoder(c_last)` — convolutional encoder that produces `z` and returns the bottleneck shape.
  - `IMAGE_SIZE = 1024` — input preprocessing size.
  - Network send: header + raw `float32` bytes of interleaved I/Q vector.
- `decoder.py`:
  - Class `Decoder(c_last, img_channels=3)` — ConvTranspose-based decoder. Use `bottleneck_shape = (C_LAST, 256, 256)` before calling `forward`.
  - `receive_all(sock, n)` — helper to ensure the full payload is read from the socket.

**Notes & Troubleshooting**

- Firewall / Connection errors: If you see `ConnectionRefusedError` from `encoder.py`, verify the decoder is running and reachable at the configured `HOST`/`PORT` and that OS firewalls allow the connection.
- Model loading: Both scripts load `.pth` files using `torch.load(..., map_location=DEVICE)`. If you do not have a GPU, the code will fall back to CPU automatically.
- Payload sizing / shapes: If the decoder fails because the reshaped tensor has the wrong size, confirm that `C_LAST`, `IMAGE_SIZE`, and expected `bottleneck_shape` align between `encoder.py` and `decoder.py`.
- Image display: The decoder uses `matplotlib.pyplot.show()` to display the reconstructed image — this may be blocking depending on your environment. Use remote display or save the output with `PIL.Image.save()` if needed.

**Possible improvements**

- Add CLI arguments for `HOST`, `PORT`, `IMAGE_PATH`, and `MODEL_PATH` instead of hard-coded constants.
- Add basic checksum or CRC in the header to validate payload integrity.
- Add a small wrapper to transmit metadata (shape, dtype) instead of relying on hard-coded shapes.
- Add unit tests and a small script to run a local encode→decode loop for quick validation.

If you want, I can:

- Add CLI flags to both scripts and update usage examples.
- Create a small local test script that runs both `encoder` and `decoder` in-process to validate shapes.

---

Repository maintained as a simple educational demo of sending semantic (latent) vectors over a socket and reconstructing images with a decoder.
