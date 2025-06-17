import numpy as np
from PIL import Image
import librosa
import math
import matplotlib.pyplot as plt
import soundfile as sf

audio_data, sample_rate = librosa.load("Take Five.wav")
length = len(audio_data)


def optimal_dimensions(size: int):
    #Derivation: SO
    min_product = float("inf")
    best_pair = (None, None)
    start = int(math.isqrt(size))
    for m in range(start, size + 1):
        n = (size + m - 1) // m  # ceil(x / m)
        product = m * n
        if product < min_product:
            min_product = product
            best_pair = (m, n)
        if (
            m > size // n
        ):  # m*n already exceeds x and increasing m further will increase the product
            break

    return best_pair


# Let's think about this. The most brain-dead solution is to remap floats to ints, but that likely wouldn't work due to the nonlinear nature of hearing.
# It would likely introduce strong(er) quantization issues 
# Instead, taking the log would allow for more range possibilities
# in the sound. We'll use 24 bits, as that matches nicely with both a common PCM width and 3*8=24 for RGB values
# taking the log also makes the data a nice normal distribution (apart from the padding value), which must be good for something

def unpack_rgb(packed: np.ndarray) -> np.ndarray:
    # Ensure unsigned for bitwise ops
    packed = packed.astype(np.uint32)
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    # Stack along a new last axis
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def encode_audio_to_image(audio_data: np.ndarray) -> Image.Image:
    dimensions = optimal_dimensions(length)
    assert dimensions[0] != None
    target_size = dimensions[0] * dimensions[1]
    pad_len = target_size - len(audio_data)
    if pad_len < 0:
        raise ValueError("Audio data longer than computed image size")

    safe_data = np.clip(audio_data, 1e-8, None)
    log_data = np.log10(safe_data)
    max_data = np.max(log_data)
    min_data = np.min(log_data)
    print(min_data)
    print(max_data)
    max_value = 2**24 - 1
    norm_data = (log_data - min_data) / (max_data - min_data)
    scaled_data: np.ndarray = norm_data * max_value
    plt.figure(figsize=(10, 6))
    plt.hist(scaled_data, bins=100, color="skyblue", edgecolor="black")
    plt.title("Histogram of Scaled Audio Data Values (Log-normalized to 24-bit range)")
    plt.xlabel("24-bit Scaled Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plot.png")
    if pad_len > 0:
        scaled_data = np.pad(
            scaled_data, (0, pad_len), mode="constant", constant_values=0
        )
    image_data = np.reshape(scaled_data, dimensions).astype(np.uint32)

    rgb_data = unpack_rgb(image_data)
    return Image.fromarray(rgb_data)


def pack_rgb(rgb_array: np.ndarray) -> np.ndarray:
    """
    Reverse of `unpack_rgb`: convert (H, W, 3) RGB array into 24-bit integers.
    """
    r = rgb_array[..., 0].astype(np.uint32)
    g = rgb_array[..., 1].astype(np.uint32)
    b = rgb_array[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def decode_image_to_audio(
    image: Image.Image, min_log: float, max_log: float, original_length: int
) -> np.ndarray:
    """
    Decodes an RGB image (produced by encode_audio_to_image) back to mono audio.

    Parameters:
        image: PIL Image, RGB-encoded audio data
        min_log: minimum of log-transformed audio data used in encoding
        max_log: maximum of log-transformed audio data used in encoding
        original_length: original number of audio samples before padding

    Returns:
        Reconstructed mono audio as a NumPy array
    """
    rgb_array = np.array(image)
    packed = pack_rgb(rgb_array)

    # Normalize to [0, 1] and rescale to log10 range
    norm_data = packed.astype(np.float32) / (2**24)
    log_data = norm_data * (max_log - min_log) + min_log

    # Reverse the log transform
    reconstructed_audio = 10**log_data
    print(np.min(reconstructed_audio))
    print(np.max(reconstructed_audio))
    
    # reconstructed_audio=log_data
    reconstructed_audio = reconstructed_audio.flatten()

    # Trim to original length (remove padding)
    return reconstructed_audio[:original_length]


v = encode_audio_to_image(audio_data)
file="Data.png"
# file="Data.jpg"
# v.save(file,quality=90,subsampling=0,optimize=True)
v.save(file)

new_v=Image.open(file)
# returns values between 0 and 1, rescale to -1,1
decoded_audio = decode_image_to_audio(new_v, -8, -0.03690627, length) * 2 - 1
sf.write("out.wav", decoded_audio, sample_rate, format="WAV", subtype="FLOAT")
