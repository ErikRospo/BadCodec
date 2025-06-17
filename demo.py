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

def encode_and_save(audio_data, sample_rate, fmt, img_kwargs=None, out_suffix=""):
    img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
    img_path = os.path.join(DEMO_DIR, f"encoded_audio{out_suffix}.{fmt}")
    save_kwargs = {"format": fmt.upper()}
    if img_kwargs:
        save_kwargs.update(img_kwargs)
    img.save(img_path, **save_kwargs)
    print(f"Saved {img_path}")

    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, audio_data.shape[0]
    )
    out_wav = os.path.join(DEMO_DIR, f"decoded_output{out_suffix}_{fmt}.wav")
    sf.write(
        out_wav,
        decoded_audio.T,
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    print(f"Decoded and saved {out_wav}")

def encode_and_save_jpeg(audio_data, sample_rate, quality):
    img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
    img_path = os.path.join(DEMO_DIR, f"encoded_audio_jpeg_q{quality}.jpg")
    save_kwargs = {"format": "JPEG", "quality": quality, "subsampling": 0}
    img.save(img_path, **save_kwargs)
    print(f"Saved {img_path}")

    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, audio_data.shape[0]
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

def encode_and_save_webp(audio_data, sample_rate, quality):
    img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
    img_path = os.path.join(DEMO_DIR, f"encoded_audio_webp_q{quality}.webp")
    save_kwargs = {"format": "WEBP", "quality": quality, "lossless": False}
    img.save(img_path, **save_kwargs)
    print(f"Saved {img_path}")

    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, audio_data.shape[0]
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

def encode_and_save_webp_lossless(audio_data, sample_rate):
    img, min_c, max_c, padded_length = encode_audio_to_image(audio_data)
    img_path = os.path.join(DEMO_DIR, "encoded_audio_webp_lossless.webp")
    img.save(img_path, format="WEBP", lossless=True)
    print(f"Saved {img_path}")

    loaded_img = Image.open(img_path)
    decoded_audio = decode_image_to_audio(
        loaded_img, min_c, max_c, padded_length, audio_data.shape[0]
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

# Load stereo audio
audio_data, sample_rate = librosa.load(AUDIO_FILE, mono=False)
if audio_data.ndim == 1:
    audio_data = np.expand_dims(audio_data, axis=0)

# Encode and save in different formats (default settings)
encode_and_save(audio_data, sample_rate, "png")
    

# JPEG with different qualities and subsampling
for quality in JPEG_QUALITIES:
    encode_and_save_jpeg(audio_data, sample_rate, quality)

# WEBP with different qualities (lossy)
for quality in WEBP_QUALITIES:
    encode_and_save_webp(audio_data, sample_rate, quality)

# WEBP lossless
encode_and_save_webp_lossless(audio_data, sample_rate)