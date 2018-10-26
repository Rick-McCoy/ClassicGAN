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
        self.train_data_loader = DataLoader(args.batch_size, args.shuffle, args.num_workers, True)
        self.test_data_loader = DataLoader(args.batch_size, args.shuffle, args.num_workers, False)
    
    def run(self):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size', type=int, default=10)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--channels', type=int, default=324)
    parser.add_argument('--residual_channels', type=int, default=128)
    parser.add_argument('--dilation_channels', type=int, default=128)
    parser.add_argument('--skip_channels', type=int, default=512)
    parser.add_argument('--end_channels', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--gpus', type=list, default=[2, 3, 0])
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=32)

    args = parser.parse_args()
    if torch.cuda.device_count() == 1:
        args.gpus = [0]
        args.batch_size = 1

    trainer = Trainer(args)
    trainer.run()
