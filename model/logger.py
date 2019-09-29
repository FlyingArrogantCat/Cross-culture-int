import torch
import torchvision
from pathlib import Path
from tensorboardX import SummaryWriter
import datetime


class logger(torch.nn.Module):
    def __init__(self, path):
        super(logger, self).__init__()

        def get_date():
            date_now = str(datetime.datetime.now())[:19].replace(':', '_')
            date_now = date_now.replace('-', '_')
            date_now = date_now.replace(' ', '_')
            return date_now

        self.path = path
        date_now = get_date()

        p = Path(f'{self.path}/experiment_{date_now}/')
        p.mkdir()

        self.image_path = p / 'images'
        self.image_path.mkdir()

        self.log_path = p / 'log'
        self.log_path.mkdir()

        self.writer = SummaryWriter(log_dir=str(self.log_path))
        self.global_step = 0

    def add_scalar(self, tag_name, scalar, global_step=None):
        if global_step is not None: global_step=self.global_step
        self.writer.add_scalar(tag_name, scalar_value=scalar, global_step=global_step)

    def add_image(self, tag_name, image, global_step=None):
        if global_step is not None: global_step = self.global_step
        self.writer.add_image(tag_name, image, global_step)
