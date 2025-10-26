import os
import requests
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

class Brain2MusicDataset(Dataset):
    """
    PyTorch dataset class for the Brain2Music dataset.
    Each sample consists of fMRI data and its corresponding music features.
    """
    def __init__(self, data_dir="brain2music", split="train", transform=None):
        """
        Args:
            data_dir (str): Path to the dataset folder.
            split (str): One of ['train', 'val', 'test'].
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load the dataset (assuming HDF5 format for structured access)
        data_path = os.path.join(self.data_dir, f"{split}.h5")
        with h5py.File(data_path, 'r') as f:
            self.fmri_data = np.array(f['fmri'])  # Brain activity data
            self.music_features = np.array(f['music_features'])  # Corresponding music feature embeddings

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
        music = torch.tensor(self.music_features[idx], dtype=torch.float32)
        
        if self.transform:
            fmri = self.transform(fmri)
        
        return fmri, music  # Returning as a tuple (input, target)

def download_and_extract_brain2music(url, dest_folder="brain2music"):
    """
    Downloads and extracts the Brain2Music dataset.
    
    Args:
        url (str): Direct download link to the dataset.
        dest_folder (str): Directory to store the dataset.
    """
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "brain2music.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    # Extracting dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print("Extraction complete.")

    # Remove zip file after extraction
    os.remove(zip_path)

# Example usage
if __name__ == "__main__":
    # Provide the actual URL where the Brain2Music dataset is hosted
    DATASET_URL = "/home/gkondas/.cache/kagglehub/datasets/nishimotolab/music-caption-brain2music/versions/1"
    
    # Step 1: Download and extract
    download_and_extract_brain2music(DATASET_URL)

    # Step 2: Create dataset instances
    train_dataset = Brain2MusicDataset(split="train")
    val_dataset = Brain2MusicDataset(split="val")

    # Step 3: Create DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Step 4: Iterate through dataset (example usage)
    for fmri, music in train_loader:
        print(f"fMRI Shape: {fmri.shape}, Music Features Shape: {music.shape}")
        break  # Just checking the first batch
