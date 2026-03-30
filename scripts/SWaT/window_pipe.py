import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SWaT_Dataset(Dataset):
    def __init__(self, dataframe, window_size=60, stride=10):
        """
        Takes a pandas dataframe and turns it into PyTorch-ready sliding windows.
        window_size: How many rows (seconds) of data the model sees at once.
        stride: How many rows we slide forward to make the next window.
        """
        # 1. Separate features (the 51 sensors) from the label (is_attack)
        # Assuming your label column is named 'is_attack'
        features = dataframe.drop(columns=['is_attack']).values
        labels = dataframe['is_attack'].values
        
        # 2. Convert to Numpy arrays (Way faster than pandas for slicing)
        # We use float32 because GPUs love 32-bit math!
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        
        self.window_size = window_size
        self.stride = stride
        
        # Calculate how many total windows we can fit
        self.num_windows = (len(self.features) - self.window_size) // self.stride + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        # Calculate the start and end row for this specific window
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Slice the data!
        window_data = self.features[start_idx:end_idx]
        
        # For the label, if ANY row in this window is an attack, 
        # we label the whole window as an ATTACK (1.0)
        window_label = np.max(self.labels[start_idx:end_idx])
        
        # Convert to PyTorch tensors
        return torch.tensor(window_data), torch.tensor(window_label)

# ==========================================
# 🚀 HOW TO USE THIS IN COLAB
# ==========================================

def create_dataloaders(df, window_size=60, batch_size=32):
    """
    Splits the data chronologically (No random shuffling!) 
    Train: 65%, Val: 15%, Test: 20%
    """
    print("🔪 Chopping data chronologically...")
    n = len(df)
    train_df = df[0 : int(n * 0.65)]
    val_df   = df[int(n * 0.65) : int(n * 0.80)]
    test_df  = df[int(n * 0.80) :]
    
    # Create the Dataset objects
    # Note: Train has a smaller stride to generate MORE training examples
    train_dataset = SWaT_Dataset(train_df, window_size=window_size, stride=10)
    val_dataset   = SWaT_Dataset(val_df, window_size=window_size, stride=window_size)
    test_dataset  = SWaT_Dataset(test_df, window_size=window_size, stride=window_size)
    
    # Wrap them in DataLoaders (This feeds the data to the GPU in batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Generated {len(train_dataset)} training windows!")
    return train_loader, val_loader, test_loader

# Example:
# train_loader, val_loader, test_loader = create_dataloaders(my_swat_dataframe)