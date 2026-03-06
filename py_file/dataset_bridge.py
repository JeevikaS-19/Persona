import torch
import pandas as pd
import cv2
from torch.utils_data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class DeepfakeSequenceDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Forensic Sequences.
    Groups 16 frames from the same source to allow GRU temporal training.
    """
    def __init__(self, csv_file, sequence_length=16, transform=None):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Group by filenames to identify sequences (assumes filenames contain unique video IDs)
        # In a real scenario, we might need a 'video_id' column. 
        # For now, we'll group by member and chronological order.
        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self):
        # Sort values to ensure chronological sequence if indexed
        self.data = self.data.sort_values(by=['team_member_id', 'filename'])
        
        sequences = []
        # Basic logical grouping: Every 16 frames from the same team_member_id 
        # (assuming they were recorded in order)
        for name, group in self.data.groupby('team_member_id'):
            for i in range(0, len(group) - self.sequence_length + 1, self.sequence_length):
                seq_rows = group.iloc[i : i + self.sequence_length]
                sequences.append(seq_rows)
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_df = self.sequences[idx]
        
        frames = []
        labels = []
        
        for _, row in seq_df.iterrows():
            img_path = row['absolute_path']
            label = 1 if row['label'] in ['DEEPFAKE', 1] else 0
            
            # Load Image
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
                labels.append(label)
            except Exception as e:
                # Handle missing image during training
                frames.append(torch.zeros(3, 224, 224))
                labels.append(0)
        
        # Stack frames into [SeqLength, Channels, H, W]
        # Label is usually the label of the last frame or majority
        sequence_tensor = torch.stack(frames)
        final_label = torch.tensor(labels[-1], dtype=torch.float32)
        
        return sequence_tensor, final_label

# Example Usage:
# dataset = DeepfakeSequenceDataset("master_dataset.csv")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
