import os
import numpy as np
from PIL import Image
import librosa
import soundfile as sf
from main import encode_audio_to_image, decode_image_to_audio

DEMO_DIR = "./demo_files"
os.makedirs(DEMO_DIR, exist_ok=True)

AUDIO_FILE = "Take Five.wav"
FORMATS = ["png", "webp"]

# JPEG quality and subsampling settings
JPEG_QUALITIES = [100, 95, 85, 75]

# WEBP quality settings (lossy)
WEBP_QUALITIES = [80, 85, 90, 95, 100]

# Load stereo audio
audio_data, sample_rate = librosa.load(AUDIO_FILE, mono=False)
num_channels, length = audio_data.shape

# Encode and save in different formats (default settings)
for fmt in FORMATS:
    img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
    img_path = os.path.join(DEMO_DIR, f"encoded_audio.{fmt}")
    img.save(img_path, format=fmt.upper())
    print(f"Saved {img_path}")

    # Load image and decode
    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, num_channels
    )
    out_wav = os.path.join(DEMO_DIR, f"decoded_output_{fmt}.wav")
    sf.write(
        out_wav,
        decoded_audio.T,
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    print(f"Decoded and saved {out_wav}")

# JPEG with different qualities and subsampling
img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
for quality in JPEG_QUALITIES:
    img_path = os.path.join(DEMO_DIR, f"encoded_audio_jpeg_q{quality}.jpg")
    save_kwargs = {"format": "JPEG", "quality": quality, "subsampling": 0}

    img.save(img_path, **save_kwargs)
    print(f"Saved {img_path}")

    # Load image and decode
    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, num_channels
    )
    out_wav = os.path.join(DEMO_DIR, f"decoded_output_jpeg_q{quality}.wav")
    sf.write(
        out_wav,
        decoded_audio.T,
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    print(f"Decoded and saved {out_wav}")

# WEBP with different qualities (lossy)
img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
for quality in WEBP_QUALITIES:
    img_path = os.path.join(DEMO_DIR, f"encoded_audio_webp_q{quality}.webp")
    save_kwargs = {"format": "WEBP", "quality": quality, "lossless": False}

    img.save(img_path, **save_kwargs)
    print(f"Saved {img_path}")

    # Load image and decode
    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, num_channels
    )
    out_wav = os.path.join(DEMO_DIR, f"decoded_output_webp_q{quality}.wav")
    sf.write(
        out_wav,
        decoded_audio.T,
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    print(f"Decoded and saved {out_wav}")

# WEBP lossless
img_path = os.path.join(DEMO_DIR, "encoded_audio_webp_lossless.webp")
img.save(img_path, format="WEBP", lossless=True)
print(f"Saved {img_path}")

loaded_img = Image.open(img_path)
decoded_audio = decode_image_to_audio(
    loaded_img, min_c, max_c, padded_length, num_channels
)
out_wav = os.path.join(DEMO_DIR, "decoded_output_webp_lossless.wav")
sf.write(
    out_wav,
    decoded_audio.T,
    sample_rate,
    format="WAV",
    subtype="FLOAT",
)
print(f"Decoded and saved {out_wav}")