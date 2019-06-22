import torch
from torch import nn
import numpy as np
from utils import flatten

import h5py


class VGGish(nn.Module):
    def __init__(self, num_bands=64, num_frames=96, embedding_size=128,
                 include_classifier=False):

        """

        :param num_bands: int, default 64
            number of bands in input mel-spectre

        :param num_frames: int, default 96
            number of mel-spectre frames

        :param embedding_size:
            if include_classifier is True, the output is a embeddings
            in embedding_size dimension

        :param include_classifier: bool
            If True the model will contain fully connected head
            that converts output of the network to embeddings.
            The input must be of fixed size for it.
            If False, the outputs of convolutional block is avereged
            through time (so the input is not necessary fixed).

        input: Tensor [batch_size, num_bands, seq_len]

        output: Tensor [batch_size, embedding_size], if include_classifier
                Tensor [batch_size, 512], else
        """
        super(VGGish, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.include_classifier = include_classifier
        if include_classifier:
            self.classifier = nn.Sequential(
                # input = (num_bands // 16) * (num_frames // 16) * 512

                nn.Linear(num_bands * num_frames * 2, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, embedding_size),
            )

    def forward(self, input):
        batch_size, num_bands, seq_len = input.size

        input = input.unsqueeze(1)
        # [batch_size, embedding=1, num_bands=64, seq_len=96]

        input = self.features(input)
        # [batch_size, embedding=512, num_bands=4, seq_len=6]

        if self.include_classifier:
            input = input.reshape(batch_size, -1)
            # [batch_size, all_neurons=12288]

            return self.classifier(input)

        return input.mean(dim=-1)


def vggish(include_classifier=False, pretrained=False):
    model = VGGish(include_classifier=include_classifier)

    if pretrained:
        # weights for tf pretrained model
        file = h5py.File('./models/vggish_audioset_weights.h5', 'r')
        weights = list(flatten(file))

        # swap bias and kernels
        weights[::2], weights[1::2] = weights[1::2], weights[::2]

        for param, weight in zip(model.parameters(), weights):
            param.data = torch.Tensor(np.array(weight).T)

    return model
