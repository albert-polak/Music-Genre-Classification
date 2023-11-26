import lightning as L
from torch.utils.data import DataLoader
import pandas as pd
import librosa
import numpy as np

from torchvision import transforms

class GTZANDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.sampling_rate = 12000
        self.duration = 30
        self.skip_duration = 15
        self.batch_size = 1
        self.sample_duration = self.duration * self.sampling_rate
        self.sample_skip_duration = self.skip_duration * self.sampling_rate

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
        return mel_db

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = pd.read_csv('./train.csv', sep=',')
        self.test = pd.read_csv('./test.csv', sep=',')
        self.val = pd.read_csv('./val.csv', sep=',')
        self.train_data = []
        self.test_data = []
        self.val_data = []
        for index, i in self.train.iterrows():
            audio_file, _ = librosa.load(self.data_dir+i['label']+'/'+i['file'], sr=self.sampling_rate)

            mel_db = self.to_mel(audio_file=audio_file)
            mel_db = mel_db[np.newaxis, ...]         

            self.train_data.append([mel_db, i['label_encoded']])

        for index, i in self.test.iterrows():
            audio_file, _ = librosa.load(self.data_dir+i['label']+'/'+i['file'], sr=self.sampling_rate)
            mel_db = self.to_mel(audio_file=audio_file)
            mel_db = mel_db[np.newaxis, ...]

            self.test_data.append([mel_db, i['label_encoded']])

        for index, i in self.val.iterrows():
            audio_file, _ = librosa.load(self.data_dir+i['label']+'/'+i['file'], sr=self.sampling_rate)
            mel_db = self.to_mel(audio_file=audio_file)
            mel_db = mel_db[np.newaxis, ...]

            self.val_data.append([mel_db, i['label_encoded']])


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)