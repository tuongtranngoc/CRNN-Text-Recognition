from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import torch
import argparse
import torch.nn.functional as F

from . import *

logger = Logger.get_logger("EXPORT")


class Exporter:
    def __init__(self, args):
        self.args = args
        __, self.id2char = map_char2id()
        logger.info("Creating model ...")
        self.model = CRNN(len(self.id2char)+1)
        self.sample = torch.randn(size=cfg['Train']['dataset']['transforms']['image_shape']).unsqueeze(0).to(self.args.device)
        logger.info(f"Creating sample with shape {self.sample.shape}")
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
        pytorch2paddle(module=self.model, save_dir=f, jit_type='trace', input_examples=[self.sample])
        return f
        
    def export_onnx(self):
        import onnx
        logger.info(f'Starting export with onnx {onnx.__version__}...')
        f = str(self.args.model_path).replace('.pth', f'.onnx')
        output_names = ['output0']
        torch.onnx.export(
            self.model,
            self.sample,
            f,
            verbose=False,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=None
        )
        model_onnx = onnx.load(f)
        onnx.save(model_onnx, f)
        return f


    def export_tensorrt(self):
        workspace = 4
        import tensorrt as trt
        f_onnx = self.export_onnx()
        f = str(self.args.model_path).replace('.pth', f'.engine')
        logger.info(f"Starting export with tensorrt {trt.__version__}...")
        logger_trt = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger_trt)
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger_trt)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f'failed to load ONNX file: {f_onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        for inp in inputs:
            logger.info(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            logger.info(f'output "{out.name}" with shape{out.shape} {out.dtype}')

        config.set_flag(trt.BuilderFlag.FP16)
        del self.model
        torch.cuda.empty_cache()

        with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
            t.write(engine.serialize())
        
        return f


    def __call__(self):
        logger.info("Begining export model ...")
        if self.args.export_format == 'torchscript':
            self.export_torchscript()
        if self.args.export_format == 'paddle':
            self.export_paddle()
        if self.args.export_format == 'onnx':
            self.export_onnx()
        if self.args.export_format == 'tensorrt':
            self.export_tensorrt()
        
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to pytorch model")
    parser.add_argument("--export_format", type=str, default="paddle", help="Support export formats: torchscript, paddle, TensorRT, ONNX")
    parser.add_argument("--device", type=str, default='cuda', help="Select device for export format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    exporter = Exporter(args)
    exporter()