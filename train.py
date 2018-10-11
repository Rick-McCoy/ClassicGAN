from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
from tqdm import tqdm
from model import Wavenet
from data import DataLoader
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, args):
        self.args = args
        self.wavenet = Wavenet(args.layer_size, args.stack_size, args.in_channels, args.res_channels, args.learning_rate, args.gpus)
        self.data_loader = DataLoader(self.wavenet.receptive_fields, args.batch_size, args.shuffle, args.num_workers)
        self.writer = SummaryWriter('Logs')
    
    def run(self):
        for i, sample in tqdm(enumerate(self.data_loader)):
            real = sample
            synth, loss = self.wavenet.train(sample, real)
            tqdm.write('Step {}/{} Loss: {}'.format(i, self.args.num_steps, loss.item()))
            self.writer.add_scalar('Loss', loss)
            if i % 20 == 19:
                self.writer.add_image('Real', torch.nn.functional.pad(real, (0, 0, 0, 0, 0, 2), "constant", 0))
                self.writer.add_image('Fake', torch.nn.functional.pad(synth, (0, 0, 0, 0, 0, 2), "constant", 0))

        self.wavenet.save(args.num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=10)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=768)
    parser.add_argument('--res_channels', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--gpus', type=tuple, default=(2, 3))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run()
