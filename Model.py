import os
import torch
import torch.optim
import random
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from data import INPUT_LENGTH, clean, piano_rolls_to_midi, save_roll
from network import Wavenet as WavenetModule

class Wavenet:
    def __init__(
            self, 
            layer_size, 
            stack_size, 
            channels, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels, 
            lr, gpus, writer
        ):
        self.net = WavenetModule(
            layer_size, 
            stack_size, 
            channels, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels
        )
        self.gpus = gpus
        self._prepare_for_gpu()
        self.channels = channels
        self.lr = lr
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = self._loss()
        self.optimizer = self._optimizer()
        self.writer = writer
    
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

    def train(self, x, step):
        output = self.net(x).transpose_(1, 2)[:, :-1, :]
        real = x.transpose_(1, 2)[:, 1:, :]
        loss = self.loss(output.reshape(-1, self.channels), real.reshape(-1, self.channels))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if step % 20 == 0:
            real_image = real[:1]
            self.writer.add_image('Real', real_image, step)
            output_image = output[:1]
            self.writer.add_image('Generated', output_image, step)
        return loss.item()

    def sample(self, step):
        if not os.path.isdir('Samples'):
            os.mkdir('Samples')
        roll = self.generate_slow()
        roll = clean(roll)
        save_roll(roll, step)
        midi = piano_rolls_to_midi(roll)
        midi.write('Samples/{}.mid'.format(step))
        tqdm.write('Save to Samples/{}.mid'.format(step))
        roll = np.expand_dims(roll.T, axis=0)
        return roll

    def generate_with_input(self, input):
        output = self.net(input)
        return output

    def generate_slow(self):
        channels = [0, 72, 48, 72, 48, 48, 36]
        for i in range(1, 7):
            channels[i] += channels[i - 1]
        output = np.zeros([1, self.channels, 1])
        if random.randint(0, 1) == 1:
            output[:, 0, :] = 1
        else:
            for i in range(6):
                if random.randint(0, 1) == 1:
                    output[:, random.randint(channels[i], channels[i + 1] - 1), :] = 1
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
