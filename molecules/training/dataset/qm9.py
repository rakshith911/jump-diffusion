import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Dataset Preparation and Feature Extraction
class E_ScooterDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.scaler = StandardScaler()

        # Preprocess and normalize the dataset
        self.data['StartDate'] = pd.to_datetime(self.data['StartDate'])
        self.lat_long_data = self.data[['StartLatitude', 'StartLongitude', 'EndLatitude', 'EndLongitude']]
        self.normalized_data = self.scaler.fit_transform(self.lat_long_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'start_lat': self.normalized_data[idx, 0],
            'start_long': self.normalized_data[idx, 1],
            'end_lat': self.normalized_data[idx, 2],
            'end_long': self.normalized_data[idx, 3],
            'start_time': self.data.iloc[idx]['StartTime'],
            'trip_duration': self.data.iloc[idx]['TripDuration']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_lat_long_points(self):
        return self.lat_long_data

# Collation and Configuration
def collate_fn(batch):
    start_lats = torch.tensor([item['start_lat'] for item in batch], dtype=torch.float32)
    start_longs = torch.tensor([item['start_long'] for item in batch], dtype=torch.float32)
    end_lats = torch.tensor([item['end_lat'] for item in batch], dtype=torch.float32)
    end_longs = torch.tensor([item['end_long'] for item in batch], dtype=torch.float32)
    trip_durations = torch.tensor([item['trip_duration'] for item in batch], dtype=torch.float32)
    start_times = [item['start_time'] for item in batch]  # Keep start times as strings for now

    return start_lats, start_longs, end_lats, end_longs, trip_durations, start_times

def load_dataloader(csv_file, batch_size=32, shuffle=True):
    dataset = E_ScooterDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

# Visualization (optional)
def visualize_data(dataset):
    lat_long_points = dataset.get_lat_long_points()

    plt.scatter(lat_long_points['StartLongitude'], lat_long_points['StartLatitude'], label='Start Points', alpha=0.5)
    plt.scatter(lat_long_points['EndLongitude'], lat_long_points['EndLatitude'], label='End Points', alpha=0.5, color='red')
    plt.legend()
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('E-Scooter Trip Start and End Points')
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Replace with the path to your local CSV file
    csv_file = '/data/ChristianShelton/rmahishi/DocklessTripOpenData_9.csv'
    
    # Load and inspect data
    dataloader = load_dataloader(csv_file)
    
    for batch in dataloader:
        start_lats, start_longs, end_lats, end_longs, trip_durations, start_times = batch
        print(f"Batch of start latitudes: {start_lats}")
        print(f"Batch of trip durations: {trip_durations}")
        break  # Only show the first batch for brevity

    # Visualization (optional)
    visualize_data(dataloader.dataset)
