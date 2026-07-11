import random
import torch
import torchvision.transforms as T
import torchaudio.transforms as AT

# ==========================================
# Visual Augmentations
# ==========================================
def get_visual_train_transforms(image_size=224):
    """
    Returns data augmentation transforms for face frames during training.
    """
    return T.Compose([
        T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        T.RandomCrop((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_visual_val_transforms(image_size=224):
    """
    Returns standard transforms for face frames during validation/testing.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ==========================================
# Audio Augmentations
# ==========================================
def add_audio_noise(waveform, noise_level=0.005):
    """Adds random Gaussian noise to the audio waveform."""
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def apply_audio_augmentations(waveform, sample_rate=16000):
    """Applies a random set of audio augmentations."""
    if random.random() < 0.5:
        waveform = add_audio_noise(waveform, noise_level=random.uniform(0.001, 0.01))
    
    # Random gain
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        waveform = waveform * gain
    
    return waveform

# ==========================================
# Text Augmentations
# ==========================================
def random_word_dropout(text, p=0.1):
    """Randomly drops words from text with probability p."""
    words = text.split()
    if len(words) < 3:
        return text
    
    kept_words = [w for w in words if random.random() > p]
    if len(kept_words) == 0:
        kept_words = [random.choice(words)]
        
    return " ".join(kept_words)

def apply_text_augmentations(text):
    """Applies a random set of text augmentations."""
    if random.random() < 0.3:
        text = random_word_dropout(text, p=0.1)
    return text
