import torch
import os
from tensorboardX import SummaryWriter


# Config 파일 수정
# dataset_root 를 수정할 꼐획.

class Config():
    '''
    Config class
    '''
    def __init__(self,args):
        # datasets root
        self.train_dataset_root = [
            args.train_image_root,
            args.train_image_gt_root,
            args.train_image_density_root
        ]
        self.test_dataset_root = [
            args.test_image_root,
            args.test_image_gt_root,
            args.test_image_density_root
        ]
        self.device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr           = args.lr               # learning rate
        self.batch_size   = args.batch_size                   # batch size
        self.epochs       = args.epoch                # epochs
        self.checkpoints  = './checkpoints'     # checkpoints dir
        self.writer       = SummaryWriter()     # tensorboard writer

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)