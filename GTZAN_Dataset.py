import lightning as L
from torch.utils.data import DataLoader
import pandas as pd
import librosa
import numpy as np
# Note - you must have torchvision installed for this example

from torchvision import transforms

class GTZANDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.sampling_rate = 12000
        self.duration = 30
        self.skip_duration = 15
        self.sample_duration = self.duration * self.sampling_rate
        self.sample_skip_duration = self.skip_duration * self.sampling_rate

    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def to_mel(self, audio_file):
        audio_file = audio_file[self.sample_skip_duration:self.sample_duration+self.sample_skip_duration]

        if audio_file.shape[0] < self.sample_duration:
            src = np.hstack((src, np.zeros((int(self.sample_duration) - audio_file.shape[0],))))
        
        mel = librosa.feature.melspectrogram(y=audio_file, sr=self.sampling_rate)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        return mel_db

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train = pd.read_csv('./train.csv')
        self.test = pd.read_csv('./test.csv')
        self.val = pd.read_csv('./val.csv')

        self.train_data = []
        self.test_data = []
        self.val_data = []
        for i in self.train:
            audio_file = librosa.load(self.data_dir+i[0], sr=self.sampling_rate)

            mel_db = self.to_mel(audio_file=audio_file)            

            self.train_data.append([mel_db, i[1]])

        for i in self.test:
            audio_file = librosa.load(self.data_dir+i[0], sr=self.sampling_rate)
            mel_db = self.to_mel(audio_file=audio_file)

            self.test_data.append([mel_db, i[1]])

        for i in self.val:
            audio_file = librosa.load(self.data_dir+i[0], sr=self.sampling_rate)
            mel_db = self.to_mel(audio_file=audio_file)

            self.val_data.append([mel_db, i[1]])


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)