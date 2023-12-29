import torch
from torch import nn

from torch.optim import Adam
from torch.distributions.uniform import Uniform

import lightning as L

class WordEmbeddingFromScratch(L.LightningModule):
    def __init__(self):
        super().__init__()

        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        
        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x[0]

        input_to_top_hidden = (
            x[0] * self.input1_w1
            + x[1] * self.input2_w1
            + x[2] * self.input3_w1
            + x[3] * self.input4_w1
        )

        input_to_bottom_hidden = (
            x[0] * self.input1_w2
            + x[1] * self.input2_w2
            + x[2] * self.input3_w2
            + x[3] * self.input4_w2
        )

        output_1 = (
            input_to_top_hidden * self.output1_w1
            + input_to_bottom_hidden * self.output1_w2
        )

        output_2 = (
            input_to_top_hidden * self.output2_w1
            + input_to_bottom_hidden * self.output2_w2
        )

        output_3 = (
            input_to_top_hidden * self.output3_w1
            + input_to_bottom_hidden * self.output3_w2
        )

        output_4 = (
            input_to_top_hidden * self.output4_w1
            + input_to_bottom_hidden * self.output4_w2
        )

        output_presoftmax = torch.stack([output_1, output_2, output_3, output_4])

        return output_presoftmax

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx): 
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i[0])
        return loss