import os
import torch
import torch.optim
from network import Wavenet as WavenetModule

class Wavenet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr, gpus=(2, 3)):
        self.net = WavenetModule(layer_size, stack_size, in_channels, res_channels)
        self.receptive_fields = self.net.receptive_field
        self.in_channels = in_channels
        self.lr = lr
        self.gpus = gpus
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = self._loss()
        self.optimizer = self._optimizer()
    
    @staticmethod
    def _loss():
        loss = torch.nn.BCEWithLogitsLoss()
        
        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _prepare_for_gpu(self):
        if len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus)

        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, input, real):
        output = self.net(input)
        loss = self.loss(output.view(-1, self.in_channels), real.long().view(-1, self.in_channels))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def generate(self, input):
        output = self.net(input)
        return self.sigmoid(output)

    def save(self, step):
        torch.save(self.net.state_dict(), 'Checkpoints/{}.pkl'.format(step))
    
    def load(self, step):
        self.net.load_state_dict(torch.load('Checkpoints/{}.pkl'.format(step)))
