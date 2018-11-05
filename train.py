from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
import pathlib
import time
from tqdm import tqdm
from model import Wavenet
from data import DataLoader, natural_sort_key
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, args):
        self.args = args
        self.train_writer = SummaryWriter('Logs/train')
        self.test_writer = SummaryWriter('Logs/test')
        self.wavenet = Wavenet(
            args.layer_size, 
            args.stack_size, 
            args.channels, 
            args.residual_channels, 
            args.dilation_channels, 
            args.skip_channels, 
            args.end_channels, 
            args.learning_rate, 
            args.gpus, 
            self.train_writer
        )
        self.train_data_loader = DataLoader(args.batch_size * len(args.gpus), args.shuffle, args.num_workers, True)
        self.test_data_loader = DataLoader(args.batch_size * len(args.gpus), args.shuffle, args.num_workers, False)
    
    def run(self):
        checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
        checkpoint_list = [str(i) for i in checkpoint_list]
        if len(checkpoint_list) > 0:
            checkpoint_list.sort(key=natural_sort_key)
            self.wavenet.load(str(checkpoint_list[-1]))
        for epoch in tqdm(range(self.args.num_epochs)):
            for i, sample in tqdm(enumerate(self.train_data_loader), total=self.train_data_loader.__len__()):
                step = i + epoch * self.train_data_loader.__len__()
                self.wavenet.train(sample.cuda(self.args.gpus[0]), step, True, self.args.num_epochs * self.train_data_loader.__len__())
            train_loss = 0
            for i, sample in tqdm(enumerate(self.test_data_loader), total=self.test_data_loader.__len__()):
                train_loss += self.wavenet.train(sample.cuda(self.args.gpus[0]), train=False)
            train_loss /= self.test_data_loader.__len__()
            tqdm.write('Testing step Loss: {}'.format(train_loss))
            end_step = (epoch + 1) * self.train_data_loader.__len__()
            sampled_image = self.wavenet.sample(end_step)
            self.test_writer.add_scalar('Testing loss', train_loss, end_step)
            self.test_writer.add_image('Sampled', sampled_image, end_step)
            self.wavenet.save(end_step)

    def sample(self, num):
        checkpoint_list = list(pathlib.Path('Checkpoints').glob('**/*.pkl'))
        checkpoint_list = [str(i) for i in checkpoint_list]
        if len(checkpoint_list) > 0:
            checkpoint_list.sort(key=natural_sort_key)
            self.wavenet.load(str(checkpoint_list[-1]))
        for i, sample in tqdm(enumerate(self.test_data_loader), total=num):
            if i >= num:
                return
            self.wavenet.sample('Sample_{}'.format(int(time.time())), self.args.temperature, sample)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=10)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--channels', type=int, default=324)
    parser.add_argument('--residual_channels', type=int, default=256)
    parser.add_argument('--dilation_channels', type=int, default=256)
    parser.add_argument('--skip_channels', type=int, default=512)
    parser.add_argument('--end_channels', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--gpus', type=list, default=[1, 2, 3])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.)

    args = parser.parse_args()
    if torch.cuda.device_count() == 1:
        args.gpus = [0]
        args.batch_size = 1

    if args.sample > 0:
        args.gpus = [2, 3]
        args.batch_size = 1

    trainer = Trainer(args)

    if args.sample > 0:
        trainer.sample(args.sample)
    else:
        trainer.run()
