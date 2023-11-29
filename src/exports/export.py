from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import torch
import argparse
from datetime import datetime
import torch.nn.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile

from . import *

logger = Logger.get_logger("EXPORT")


class Export:
    def __init__(self, args):
        self.args = args
        __, self.id2char = map_char2id()
        self.model = CRNN(len(self.id2char)+1)
        self.sample = torch.randn(size=cfg['Train']['dataset']['transforms']['image_shape']).unsqueeze(0).to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.to(self.args.device)
        self.model.eval()
    
    def export_torchscript(self):
        f = str(self.args.model_path).replace('.pth', f'.torchscript')
        logger.info(f'Starting export with torch {torch.__version__}...')
        ts = torch.jit.trace(self.model, self.sample, strict=False)
        logger.info(f'Optimizing for mobile...')
        ts.save(f)  # https://pytorch.org/tutorials/recipes/script_optimized.html
        return f
    
    def export_paddle(self):
        import x2paddle
        from x2paddle.convert import pytorch2paddle
        logger.info(f'Starting export with X2Paddle {x2paddle.__version__}...')
        f = str(self.args.model_path).replace('.pth', f'_paddle_model{os.sep}')
        pytorch2paddle(module=self.model, save_dir=f, jit_type='trace', input_examples=[self.sample])  # export
        return f
        
    def export_onnx(self):
        pass

    def export_tensorrt(self):
        pass

    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to pytorch model")
    parser.add_argument("--export_format", type=str, default="torchscript", help="Support export formats: torchscript, paddle, TensorRT, ONNX")
    parser.add_argument("--device", type=str, default='cuda', help="Select device for export format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    exporter = ExportTool(args)
    exporter.export_paddle()