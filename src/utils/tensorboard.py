from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from . import cfg
from torch.utils.tensorboard import SummaryWriter

class Tensorboard:
    outdir = cfg['Debug']['tensorboard']
    writer = SummaryWriter(outdir)
    
    @classmethod
    def add_scalars(cls, tag, step, **kwargs):
        for k, v in kwargs.items():
            cls.writer.add_scalar(f'{tag}/{k}', v, step)
    
    @classmethod
    def add_debug_images(cls, tag, image, bboxes, labels, step):
        cls.writer.add_image_with_boxes(tag, image, box_tensor=bboxes, global_step=step, labels=labels)
    
    @classmethod
    def add_histogram(cls):
        pass
    
    @classmethod
    def add_figures(cls):
        pass