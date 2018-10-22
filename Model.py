import os
import torch
import torch.optim
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from data import INPUT_LENGTH, clean, piano_rolls_to_midi
from network import Wavenet as WavenetModule

class Wavenet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr, gpus):
        self.net = WavenetModule(layer_size, stack_size, in_channels, res_channels)
        self.gpus = gpus
        self._prepare_for_gpu()
        self.in_channels = in_channels
        self.lr = lr
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = self._loss()
        self.optimizer = self._optimizer()
    
    def _loss(self):
        loss = torch.nn.BCELoss()
        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _prepare_for_gpu(self):
        if len(self.gpus) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.gpus, output_device=self.gpus[0])
        if torch.cuda.is_available():
            self.net.cuda(self.gpus[0])

    def train(self, input, real):
        output = self.net(input)[:, :, :-1]
        output = output.transpose(1, 2)
        real = real[:, 1:, :]
        real = real.contiguous().view(-1, self.in_channels).cuda(self.gpus[0])
        loss = self.loss(output.contiguous().view(-1, self.in_channels), real)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def sample(self, step):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate_slow()
        roll = clean(roll)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Save to Samples/{}.mid'.format(step))

    def generate_with_input(self, input):
        output = self.net(input)
        return output

    def generate_slow(self):
        output = np.zeros([1, 768, 1])
        for i in range(6):
            output[:, i * 128, :] = 1
        for i in tqdm(range(INPUT_LENGTH - 1)):
            x = torch.Tensor(output[:, :, -1024:]).cuda(self.gpus[0])
            x = self.net(x)[:, :, -1:].detach().cpu().numpy()
            output = np.concatenate((output, x), axis=2)
            del x
        return output

    def save(self, step):
        if not os.path.exists('Checkpoints'):
            os.mkdir('Checkpoints')
        torch.save(self.net.state_dict(), 'Checkpoints/{}.pkl'.format(step))
    
    def load(self, step):
        self.net.load_state_dict(torch.load('Checkpoints/{}.pkl'.format(step)))
