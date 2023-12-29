from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform

import lightning as L


class WordEmbeddingWithLinear(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x[0]

        hidden = self.input_to_hidden(x)
        output_values = self.hidden_to_output(hidden)

        return output_values

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss
