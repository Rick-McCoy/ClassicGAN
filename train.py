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
        self.wavenet = Wavenet(
            args.layer_size, 
            args.stack_size, 
            args.in_channels, 
            args.res_channels, 
            args.learning_rate, 
            args.gpus
        )
        self.data_loader = DataLoader(args.batch_size, args.shuffle, args.num_workers)
        self.writer = SummaryWriter('Logs')
    
    def run(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            for i, sample in tqdm(enumerate(self.data_loader), total=self.data_loader.__len__()):
                step = i + epoch * self.data_loader.__len__()
                real = sample.transpose(1, 2)
                synth, loss = self.wavenet.train(sample, real)
                tqdm.write('Step {}/{} Loss: {}'.format(step, self.args.num_epochs, loss))
                self.writer.add_scalar('Loss', loss, step)
                if step % 20 == 0:
                    real_image = real[:1]
                    self.writer.add_image('Real', real_image, step)
                    synth_image = synth[:1]
                    self.writer.add_image('Generated', synth_image, step)
            end_step = (epoch + 1) * self.data_loader.__len__()
            sampled_image = self.wavenet.sample(end_step)
            self.writer.add_image('Sampled', sampled_image, end_step)
            self.wavenet.save(end_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=10)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--in_channels', type=int, default=324)
    parser.add_argument('--res_channels', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--gpus', type=list, default=[2, 3, 0])
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    if torch.cuda.device_count() == 1:
        args.gpus = [0]
        args.batch_size = 1

    trainer = Trainer(args)
    trainer.run()
