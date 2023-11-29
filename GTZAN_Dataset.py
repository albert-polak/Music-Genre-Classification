import lightning as L
from torch.utils.data import DataLoader
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import random
from torchvision import transforms

class GTZANDataset(Dataset):
    def __init__(self, data_dir, dataframe, sampling_rate=12000, duration=7, skip_duration=15, mode="train"):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.skip_duration = skip_duration
        self.sample_duration = self.duration * self.sampling_rate
        self.sample_skip_duration = self.skip_duration * self.sampling_rate
        self.dataframe = dataframe
        self.mode = mode

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.dataframe.iloc[idx]['label'], self.dataframe.iloc[idx]['file'])
        audio_file, _ = librosa.load(file_path, sr=self.sampling_rate)

        if len(audio_file) >= self.sample_duration:
            start_idx = random.randint(0, len(audio_file) - self.sample_duration)
        else:
            start_idx = 0
        if self.mode == "train":
            audio_file = audio_file[start_idx:self.sample_duration + start_idx]
        

        if audio_file.shape[0] < self.sample_duration:
            audio_file = np.hstack((audio_file, np.zeros((int(self.sample_duration) - audio_file.shape[0],))))
        elif audio_file.shape[0] > self.sample_duration:
            audio_file = audio_file[:self.sample_duration]


        mel = librosa.feature.melspectrogram(y=audio_file, sr=self.sampling_rate)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        

        # Apply normalization here
        mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)
        mel_db = mel_db[np.newaxis, ...]       
        label_encoded = self.dataframe.iloc[idx]['label_encoded']

        return mel_db, label_encoded

class GTZANDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        

    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def to_mel(self, audio_file):
        audio_file = audio_file[self.sample_skip_duration:self.sample_duration+self.sample_skip_duration]

        if audio_file.shape[0] < self.sample_duration:
            audio_file = np.hstack((audio_file, np.zeros((int(self.sample_duration) - audio_file.shape[0],))))
        
        mel = librosa.feature.melspectrogram(y=audio_file, sr=self.sampling_rate)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        
        # Apply normalization here
        mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)
        
        
        return mel_db

    def setup(self, stage: str):
               # Load dataframes for train, test, and val
        self.train_df = pd.read_csv('./train.csv', sep=',')
        self.test_df = pd.read_csv('./test.csv', sep=',')
        self.val_df = pd.read_csv('./val.csv', sep=',')

        # Create dataset instances
        self.train_dataset = GTZANDataset(self.data_dir, self.train_df, mode="train")
        self.test_dataset = GTZANDataset(self.data_dir, self.test_df, mode="test")
        self.val_dataset = GTZANDataset(self.data_dir, self.val_df, mode="val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)