"""Definición de modelos para clasificación de EEG."""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet: CNN compacta para BCI (Lawhern et al., 2018).

    Arquitectura: temporal conv -> depthwise spatial conv -> separable conv -> classifier
    La capa spatial tiene kernel (n_channels, 1), cambia con cada config de electrodos.
    """

    def __init__(self, n_channels, n_times, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(F2 * (n_times // 32), n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x.flatten(1))
