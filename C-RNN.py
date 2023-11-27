from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torchvision import models

from torchvision import transforms

from torchmetrics import Accuracy  

import lightning as L
from lightning.pytorch.tuner import Tuner

from torch.optim import Adam

import librosa

from GTZAN_Dataset import GTZANDataModule

class C_RNN(L.LightningModule):
    def __init__(self, lr=1e-3, batch_size=32):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size

        self.num_classes = 10

        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3)),
            nn.ReLU(),
           
        )

        self.gru =  nn.GRU(83520, 256, 2, batch_first=True)
        self.fc = torch.nn.Linear(256, 10)

        self.loss_fn = nn.CrossEntropyLoss()  

        self.acc = Accuracy('multiclass', num_classes=self.num_classes)

        self.optimizer = Adam(self.parameters(), lr=self.lr)  

    def forward(self, X):
        X = X.to(torch.float32).cuda()
        X = self.cnn_model(X)
        # print(X.shape)
        X = X.view(X.size(0), X.size(1), X.size(2) * X.size(3))

        # print(X.shape)

        h0 = torch.randn(2, X.shape[0], 256).cuda()

        X, hn = self.gru(X, h0)

        output = X[:, -1, :]

        output = self.fc(output)

        # print(output.shape)

        return output
    
    def configure_optimizers(self):
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.90)
            return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        # y = y.squeeze()  # Remove singleton dimensions
        # print(y)
        # print(preds)

        loss = self.loss_fn(preds, y)
        # acc = self.acc(preds, y)
        acc_preds = torch.argmax(preds, dim=1)
        acc = self.acc(acc_preds, y)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)



if __name__ == "__main__":

    rnn = C_RNN()
    datamodule = GTZANDataModule('./data/genres_original/')

    save_path = "./models"

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    trainer_args = {
        "accelerator": "gpu",
        "devices": "auto",
        "strategy": "auto",
        "max_epochs": 1000,
        "callbacks": [checkpoint_callback],
        "precision": 32,
    }

    trainer = L.Trainer(**trainer_args)

    # duration = 30

    # audio_sample, sampling_rate = librosa.load('./data/genres_original/blues/blues.00000.wav') 
    # audio_sample, _ = librosa.effects.trim(audio_sample)

    # mel = librosa.feature.melspectrogram(y=audio_sample, sr=sampling_rate)
    # mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    # mel_db = mel_db[np.newaxis, ...]
    # mel_db = torch.tensor(mel_db)

    # rnn.forward(mel_db)

    trainer.fit(model=rnn, datamodule=datamodule)
