import numpy as np
from PIL import Image
import librosa
import math
import soundfile as sf

# Load stereo audio (shape: 2 x samples)
audio_data, sample_rate = librosa.load("Take Five.wav", mono=False)
num_channels, length = audio_data.shape

def optimal_dimensions(size: int):
    min_product = float("inf")
    best_pair = (None, None)
    start = int(math.isqrt(size))
    for m in range(start, size + 1):
        n = (size + m - 1) // m
        product = m * n
        if product < min_product:
            min_product = product
            best_pair = (m, n)
        if m > size // n:
            break
    return best_pair

def unpack_rgb(packed: np.ndarray) -> np.ndarray:
    packed = packed.astype(np.uint32)
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)

def pack_rgb(rgb_array: np.ndarray) -> np.ndarray:
    r = rgb_array[..., 0].astype(np.uint32)
    g = rgb_array[..., 1].astype(np.uint32)
    b = rgb_array[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b

ALPHA = 10000  # Controls compression strength

def signed_log_encode(audio: np.ndarray, alpha=ALPHA) -> np.ndarray:
    sign = np.sign(audio)
    magnitude = np.abs(audio)
    compressed = sign * np.log10(1 + alpha * magnitude)
    return compressed

def signed_log_decode(compressed: np.ndarray, alpha=ALPHA) -> np.ndarray:
    sign = np.sign(compressed)
    magnitude = np.abs(compressed)
    audio = sign * (10**magnitude - 1) / alpha
    return audio

def encode_audio_to_image(audio_data: np.ndarray) -> (Image.Image, float, float, int):
    channels, length = audio_data.shape
    audio_clipped = np.clip(audio_data, -1.0, 1.0)

    compressed = signed_log_encode(audio_clipped, ALPHA)

    interleaved = compressed.T.flatten()

    min_val = np.min(interleaved)
    max_val = np.max(interleaved)
    norm_data = (interleaved - min_val) / (max_val - min_val)

    dimensions = optimal_dimensions(len(norm_data))
    target_size = dimensions[0] * dimensions[1]
    pad_len = target_size - len(norm_data)
    if pad_len < 0:
        raise ValueError("Audio data longer than computed image size")

    max_val_int = 2**24 - 1
    scaled_data = (norm_data * max_val_int).astype(np.uint32)

    if pad_len > 0:
        scaled_data = np.pad(scaled_data, (0, pad_len), mode="constant", constant_values=0)

    image_data = scaled_data.reshape(dimensions)
    rgb_data = unpack_rgb(image_data)
    return Image.fromarray(rgb_data), min_val, max_val, target_size

def decode_image_to_audio(image: Image.Image, min_val: float, max_val: float, padded_length: int, channels: int) -> np.ndarray:
    rgb_array = np.array(image)
    packed = pack_rgb(rgb_array)

    max_val_int = 2**24 - 1
    norm_data = packed.astype(np.float32) / max_val_int
    compressed = norm_data * (max_val - min_val) + min_val

    audio_reconstructed = signed_log_decode(compressed)

    # Reshape interleaved data back to (samples, channels)
    audio_reconstructed = audio_reconstructed[:padded_length].reshape((padded_length // channels, channels))

    # Transpose to shape (channels, samples)
    return audio_reconstructed.T

# Encoding
encoded_image, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
encoded_image.save("encoded_log_audio_stereo.png")

loaded_image = Image.open("encoded_log_audio_stereo.png")
decoded_audio = decode_image_to_audio(loaded_image, min_c, max_c, padded_length, num_channels)

# Save result
sf.write("decoded_log_output_stereo.wav", decoded_audio.T, sample_rate, format="WAV", subtype="FLOAT")
